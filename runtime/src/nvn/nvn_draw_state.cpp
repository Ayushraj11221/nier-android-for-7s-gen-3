// The replay's state commands (split from nvn_draw.cpp): binds, viewports, uniform updates, counters; draws and
// dispatches go back to nvn_draw.cpp.
#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "nvn_draw_state.h"
#include "platform.h"

namespace nx::nvn {

using namespace replay;

namespace {
uint64_t g_pipeline_state_gen = 1;

struct DeferredLookupCache {
    struct Entry {
        const DeferredValues* values = nullptr;
        uint64_t address = 0;
        DeferredValues::const_iterator it{};
        bool valid = false;
    } entries[8];
};
thread_local DeferredLookupCache g_deferred_lookup;

DeferredValues::const_iterator deferred_find(const DeferredValues* values, uint64_t address) {
    auto& cache = g_deferred_lookup.entries[(address >> 4) & 7];
    if (!cache.valid || cache.values != values || cache.address != address) {
        cache.values = values;
        cache.address = address;
        cache.it = values ? values->find(address) : DeferredValues::const_iterator{};
        cache.valid = true;
    }
    return cache.it;
}

template <typename T>
bool assign_pipeline_state(T& dst, const T& src) {
    if (std::memcmp(&dst, &src, sizeof(T)) == 0) return false;
    std::memcpy(&dst, &src, sizeof(T));
    return true;
}
}  // namespace

namespace replay {
uint64_t pipeline_state_generation() { return g_pipeline_state_gen; }
void pipeline_state_changed() { ++g_pipeline_state_gen; }
}  // namespace replay

void deferred_lookup_reset() { g_deferred_lookup = {}; }

void draw_command(VkCommandBuffer cb, const Cmd& c, const CmdList& list, const Targets& targets) {
    State& s = g_state;
    auto blob = [&](size_t size) -> const uint8_t* {
        return c.blob < list.blobs.size() && list.blobs[c.blob].size() >= size ? list.blobs[c.blob].data() : nullptr;
    };
    switch (c.op) {
        case Cmd::SetViewport:
            std::memcpy(s.viewport, c.i, sizeof s.viewport), s.has_viewport = true;
            break;
        case Cmd::SetRasterizerDiscard: s.rasterizer_discard = c.n != 0; break;
        case Cmd::SetPrimitiveRestart: {
            const bool restart = c.n != 0;
            if (s.restart != restart) pipeline_state_changed();
            s.restart = restart, s.restart_index = uint32_t(c.a[0]);
            break;
        }
        case Cmd::BindImage:
            if (c.i[0] >= 0 && c.i[0] < kStages && c.i[1] >= 0 && c.i[1] < 16) s.images[c.i[0]][c.i[1]] = c.a[0];
            break;
        case Cmd::DispatchCompute: dispatch_call(cb, c); break;
        case Cmd::SetRenderEnable:
            s.render_enable = c.n != 0;
            s.cond_on = false;
            occlusion_render_enable();
            break;
        case Cmd::SetScissor:
            std::memcpy(s.scissor, c.i, sizeof s.scissor), s.has_scissor = true;
            break;
        case Cmd::SetDepthRange:
            s.depth_near = c.f[0], s.depth_far = c.f[1];
            break;
        case Cmd::BindProgram: {
            s.program = c.a[0];
            const uint32_t bits = c.mask ? c.mask : (1u << kStages) - 1;  // NVN_SHADER_STAGE_*_BIT, 1 << stage
            for (int k = 0; k < kStages; k++)
                if (bits & (1u << k)) s.stage_programs[k] = c.a[0];
            break;
        }
        case Cmd::BindUniformBuffer:
            if (c.i[0] >= 0 && c.i[0] < kStages && c.i[1] >= 0 && c.i[1] < 18) s.uniforms[c.i[0]][c.i[1]] = {c.a[0], c.a[1]};
            break;
        case Cmd::BindStorageBuffer:
            if (c.i[0] >= 0 && c.i[0] < kStages && c.i[1] >= 0 && c.i[1] < 16) s.storage[c.i[0]][c.i[1]] = {c.a[0], c.a[1]};
            break;
        case Cmd::BindDeferred: {
            const DeferredValues* dv = current_deferred();
            auto it = deferred_find(dv, c.a[0]);
            int st = c.i[0];
            if (st < 0 || st >= kStages || c.mask > 3) break;
            const auto* vec = (dv && it != dv->end()) ? &it->second : nullptr;
            const size_t vec_sz = vec ? vec->size() : 0;
            const uint32_t words = c.mask >= 2 ? 2 : 1;
            const int limit = c.mask == 0 ? 32 : c.mask == 1 ? 16 : c.mask == 2 ? 18 : 16;
            const int begin = std::max(c.i[1], 0);
            const int end = int(std::min<int64_t>(int64_t(c.i[1]) + c.n, limit));
            if (begin >= end) break;
            const uint32_t first = uint32_t(begin - c.i[1]);
            auto value = [&](uint32_t element, uint32_t word) {
                const size_t at = size_t(element) * words + word;
                return at < vec_sz ? (*vec)[at] : ld<uint64_t>(c.a[0] + 8 * at);
            };
            if (c.mask == 0) {
                for (int slot = begin; slot < end; ++slot) s.textures[st][slot] = value(first + uint32_t(slot - begin), 0);
            } else if (c.mask == 1) {
                for (int slot = begin; slot < end; ++slot) s.images[st][slot] = value(first + uint32_t(slot - begin), 0);
            } else if (c.mask == 2) {
                for (int slot = begin; slot < end; ++slot) {
                    const uint32_t element = first + uint32_t(slot - begin);
                    s.uniforms[st][slot] = {value(element, 0), value(element, 1)};
                }
            } else {
                for (int slot = begin; slot < end; ++slot) {
                    const uint32_t element = first + uint32_t(slot - begin);
                    s.storage[st][slot] = {value(element, 0), value(element, 1)};
                }
            }
            break;
        }
        case Cmd::BindTexture:  // mask 0: BindTexture, 1: BindSeparateTexture, 2: BindSeparateSampler
            if (c.i[0] < 0 || c.i[0] >= kStages || c.i[1] < 0) break;
            if (c.mask == 0 && c.i[1] < 32) s.textures[c.i[0]][c.i[1]] = c.a[0];
            if (c.mask == 1 && c.i[1] < State::kSeparateTextures) s.separate_textures[c.i[0]][c.i[1]] = c.a[0];
            if (c.mask == 2 && c.i[1] < State::kSeparateSamplers) s.separate_samplers[c.i[0]][c.i[1]] = c.a[0];
            break;
        case Cmd::BindVertexBuffer:
            if (c.i[0] >= 0 && c.i[0] < 16) s.vertex_buffers[c.i[0]] = {c.a[0], c.a[1]};
            break;
        case Cmd::SetTexturePool: s.texture_pool = c.a[0]; break;
        case Cmd::SetSamplerPool: s.sampler_pool = c.a[0]; break;
        case Cmd::BindVertexAttribs:
            if (auto* p = blob(c.n * sizeof(VertexAttrib))) {
                const size_t n = c.n;
                if (s.attribs.size() != n || (n && std::memcmp(s.attribs.data(), p, n * sizeof(VertexAttrib)) != 0)) {
                    s.attribs.assign(reinterpret_cast<const VertexAttrib*>(p), reinterpret_cast<const VertexAttrib*>(p) + n);
                    pipeline_state_changed();
                }
            }
            break;
        case Cmd::BindVertexStreams:
            if (auto* p = blob(c.n * sizeof(VertexStream))) {
                const size_t n = c.n;
                if (s.streams.size() != n || (n && std::memcmp(s.streams.data(), p, n * sizeof(VertexStream)) != 0)) {
                    s.streams.assign(reinterpret_cast<const VertexStream*>(p), reinterpret_cast<const VertexStream*>(p) + n);
                    pipeline_state_changed();
                }
            }
            break;
        case Cmd::BindBlendState:
            if (auto* p = blob(sizeof(BlendTarget))) {
                BlendTarget b;
                std::memcpy(&b, p, sizeof b);
                if (b.target < 8 && assign_pipeline_state(s.blend[b.target], b)) pipeline_state_changed();
            }
            break;
        case Cmd::BindColorState:
            if (auto* p = blob(sizeof(ColorStateInfo))) {
                ColorStateInfo value;
                std::memcpy(&value, p, sizeof value);
                if (assign_pipeline_state(s.color, value)) pipeline_state_changed();
            }
            break;
        case Cmd::BindChannelMaskState:
            if (auto* p = blob(sizeof(ChannelMaskInfo))) {
                ChannelMaskInfo value;
                std::memcpy(&value, p, sizeof value);
                if (assign_pipeline_state(s.mask, value)) pipeline_state_changed();
            }
            break;
        case Cmd::BindDepthStencilState:
            if (auto* p = blob(sizeof(DepthStencilInfo))) {
                DepthStencilInfo value;
                std::memcpy(&value, p, sizeof value);
                if (assign_pipeline_state(s.depth, value)) pipeline_state_changed();
            }
            break;
        case Cmd::BindPolygonState:
            if (auto* p = blob(sizeof(PolygonInfo))) {
                PolygonInfo value;
                std::memcpy(&value, p, sizeof value);
                if (assign_pipeline_state(s.polygon, value)) pipeline_state_changed();
            }
            break;
        case Cmd::BindMultisampleState: {
            const bool alpha_to_coverage = c.n != 0;
            if (s.alpha_to_coverage != alpha_to_coverage) pipeline_state_changed();
            s.alpha_to_coverage = alpha_to_coverage;
            break;
        }
        case Cmd::SetBlendColor: std::memcpy(s.blend_color, c.f, sizeof s.blend_color); break;
        case Cmd::UpdateUniformBuffer: {  // (buffer address, size, offset, data): written in order, like the GPU
            uint64_t off = c.a[2];
            // NX_TRACE_UUB=<first draw>: logs 40 updates from that draw on.
            static const char* uub = std::getenv("NX_TRACE_UUB");
            static const uint64_t uub_from = uub ? std::strtoull(uub, nullptr, 10) : 0;
            static int uub_n = 0;
            if (uub && g_stats.done >= uub_from && uub_n++ < 1500 && c.blob < list.blobs.size()) {
                const auto& b = list.blobs[c.blob];
                float f[8] = {};
                std::memcpy(f, b.data(), std::min<size_t>(b.size(), sizeof f));
                log("uub before draw %llu: buffer %llx size %llu offset %llu data %zu bytes: %g %g %g %g | %g %g %g %g",
                    (unsigned long long)g_stats.done, (unsigned long long)c.a[0], (unsigned long long)c.a[1],
                    (unsigned long long)off, b.size(), f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
            }
            if (c.blob < list.blobs.size()) watch_write(c.a[0] + off, list.blobs[c.blob].size(), "UpdateUniformBuffer");
            if (c.blob < list.blobs.size())
                if (uint8_t* p = const_cast<uint8_t*>(guest_bytes(c.a[0] + off))) {
                    std::memcpy(p, list.blobs[c.blob].data(), list.blobs[c.blob].size());
                    dedup_invalidate(p, list.blobs[c.blob].size());
                    written_cpu_write(p, list.blobs[c.blob].size());
                }
            break;
        }
        case Cmd::ResetCounter: occlusion_reset(cb, c, targets); break;
        case Cmd::RenderConditional: occlusion_conditional(cb, c, targets); break;
        case Cmd::ReportCounter: {  // {value, timestamp}: the work before it is done when the replay gets here
            watch_write(c.a[0], 16, "ReportCounter");
            occlusion_report(cb, c, targets);
            static uint64_t reports = 0, types_seen = 0;
            if (c.n < 64 && !(types_seen >> c.n & 1)) types_seen |= 1ull << c.n, log("nvn: first counter report of type %u", c.n);
            if (uint8_t* p = const_cast<uint8_t*>(guest_bytes(c.a[0]))) {
                // NX_FAKE_GPU_TIME=1: every report of a submit carries the submit's start (a GPU with no cost):
                // the game picks its present interval from the timings it measures (bisecting its 30 fps mode).
                static const bool fake = std::getenv("NX_FAKE_GPU_TIME") != nullptr;
                // The time since the submit's start advances at the GPU's measured rate (NX_GPU_TIME_FEED, NieR's
                // dynamic resolution; nvn_gpu_busy.cpp) or at NX_GPU_TIME_SCALE=<x> (fixed, experiments), from
                // where the last report left off when that is later: the timestamps never go back.
                static const double fixed = std::getenv("NX_GPU_TIME_SCALE") ? std::atof(std::getenv("NX_GPU_TIME_SCALE")) : 0.0;
                static uint64_t submit = 0, base = 0, last = 0;
                static double scale = 1.0;
                uint64_t now = plat::monotonic_ns();
                if (submit_start_ns() != submit) {
                    submit = submit_start_ns();
                    scale = fixed > 0 ? fixed : gpu_time_scale();
                    base = std::max(submit, last);
                }
                if (scale != 1.0) now = base + uint64_t(double(now - submit) * scale), last = std::max(last, now);
                uint64_t v[2] = {++reports, fake ? submit_start_ns() : now};
                std::memcpy(p, v, sizeof v);
                dedup_invalidate(p, sizeof v);
                written_cpu_write(p, sizeof v);
            }
            break;
        }
        case Cmd::DrawArrays: draw_call(cb, targets, c.n, false, c); break;
        case Cmd::DrawElements: draw_call(cb, targets, c.n, true, c); break;
        case Cmd::DrawElementsIndirect:
        case Cmd::DrawArraysIndirect: draw_indirect_call(cb, targets, c); break;
        case Cmd::ClearBuffer: watch_write(c.a[0], c.a[1], "ClearBuffer"); written_clear_buffer(cb, guest_bytes(c.a[0]), c.a[1], c.n); break;
        case Cmd::CopyBufferToBuffer: watch_write(c.a[1], c.a[2], ("CopyBufferToBuffer from " + std::to_string(c.a[0])).c_str()); watch_write(c.a[0], c.a[2], ("CopyBufferToBuffer source to " + std::to_string(c.a[1])).c_str()); written_copy_buffer(cb, guest_bytes(c.a[0]), guest_bytes(c.a[1]), c.a[2]); break;
        case Cmd::ReportValue:
            watch_write(c.a[0], 4, "ReportValue");
            if (uint8_t* p = const_cast<uint8_t*>(guest_bytes(c.a[0]))) {
                std::memcpy(p, &c.n, 4);
                dedup_invalidate(p, 4);
            }
            break;
        default: raster_command(c); break;
    }
}

}  // namespace nx::nvn
