// Per-submit upload reuse (docs/nvn/DEBUG.md #30). A draw's buffers are copied from guest memory into the upload
// ring when it is replayed; the same camera block, scene constants and vertex buffers were copied again for every
// draw that bound them (~130 KB per draw, 50-70 MB per frame). Within one submit guest memory only changes through
// inline uniform updates (UpdateUniformBuffer, which invalidates what it overlaps) and compute results (ranges a
// dispatch writes are read from their written copy, nvn_written.cpp, and invalidated here): a copy made once can
// serve every later draw of the submit that binds the same bytes.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "nvn_draw_state.h"

namespace nx::nvn {
uint64_t hash_memory(const uint8_t* p, uint64_t size);  // nvn_sampler.cpp
}

namespace nx::nvn::replay {

namespace {

struct Entry {
    uint64_t size = 0, padded = 0;
    Upload upload;
};
// By host address; generation stamps make submit reset O(1) and avoid per-upload tree nodes.
struct CopySlot {
    const uint8_t* p = nullptr;
    Entry entry{};
    uint64_t generation = 0;
};
std::vector<CopySlot> g_copies(2048);
size_t g_copy_count = 0;
uint64_t g_saved = 0, g_copied = 0;

// In front of the map (a tree walk per bound buffer of every draw, and most are hits): a slot per address hash with a
// copy of that address's entry in g_copies. A slot of the current generation always equals the map's entry: both are
// written together, dedup_invalidate clears the slot of each entry it erases, dedup_reset starts a new generation.
struct Front {
    const uint8_t* p = nullptr;
    uint64_t generation = 0;
    Entry entry;
};
Front g_front[1024];
uint64_t g_generation = 1;
Front& front_of(const uint8_t* p) { return g_front[(uintptr_t(p) >> 4 ^ uintptr_t(p) >> 14) & 1023]; }

size_t copy_hash(const uint8_t* p) {
    uintptr_t v = uintptr_t(p) >> 4;
    v ^= v >> 17;
    v *= uintptr_t(0xed5ad4bb);
    v ^= v >> 11;
    return size_t(v);
}

CopySlot* copy_find(const uint8_t* p) {
    const size_t mask = g_copies.size() - 1;
    size_t at = copy_hash(p) & mask;
    for (size_t probe = 0; probe < g_copies.size(); ++probe, at = (at + 1) & mask) {
        CopySlot& slot = g_copies[at];
        if (slot.generation == g_generation && slot.p == p) return &slot;
        if (slot.generation == 0 && !slot.p) return nullptr;
    }
    return nullptr;
}

void copy_rehash(size_t capacity) {
    std::vector<CopySlot> fresh(capacity);
    const size_t mask = capacity - 1;
    for (const CopySlot& old : g_copies) {
        if (old.generation != g_generation) continue;
        size_t at = copy_hash(old.p) & mask;
        while (fresh[at].generation == g_generation) at = (at + 1) & mask;
        fresh[at] = old;
    }
    g_copies.swap(fresh);
}

CopySlot& copy_store(const uint8_t* p, const Entry& entry) {
    if (CopySlot* slot = copy_find(p)) {
        slot->entry = entry;
        return *slot;
    }
    if ((g_copy_count + 1) * 10 >= g_copies.size() * 7) copy_rehash(g_copies.size() * 2);
    const size_t mask = g_copies.size() - 1;
    size_t at = copy_hash(p) & mask, stale = g_copies.size();
    for (size_t probe = 0; probe < g_copies.size(); ++probe, at = (at + 1) & mask) {
        CopySlot& slot = g_copies[at];
        if (slot.generation == g_generation && slot.p == p) {
            slot.entry = entry;
            return slot;
        }
        if (slot.generation != g_generation && stale == g_copies.size()) stale = at;
        if (slot.generation == 0 && !slot.p) {
            if (stale != g_copies.size()) at = stale;
            CopySlot& dst = g_copies[at];
            dst.p = p, dst.entry = entry, dst.generation = g_generation;
            ++g_copy_count;
            return dst;
        }
    }
    // The table is kept below 70% current occupancy; this is only reachable after unusual stale-slot churn.
    if (stale != g_copies.size()) {
        CopySlot& dst = g_copies[stale];
        dst.p = p, dst.entry = entry, dst.generation = g_generation;
        ++g_copy_count;
        return dst;
    }
    copy_rehash(g_copies.size() * 2);
    return copy_store(p, entry);
}

// NX_UPLOAD_STATS: bytes copied per kind, and of those the bytes equal to the last copy of the same address and size.
struct KindStats {
    uint64_t copies = 0, bytes = 0, same = 0;
};
KindStats g_kind[kUploadKinds];
std::unordered_map<const uint8_t*, std::pair<uint64_t, uint64_t>> g_last_hash;  // address -> (size, hash)

// Always on (a few adds per copy): the bytes this submit copied into the ring, per kind. The spill lines of
// nvn_frames.cpp say what a huge submit is made of (dedup_submit_breakdown).
uint64_t g_submit_kind_bytes[kUploadKinds] = {};

// NX_UPLOAD_STATS: how the copies are sized, and how much a cache of 4 KB pages (instead of whole ranges) could save.
// A page (guest address, 4 KB, whole pages inside a copy only) is: repeated inside the submit (an overlapping range
// already copied it), the same as the last submit that copied it, changed since, or seen for the first time.
constexpr int kSizeBuckets = 5;  // <= 4 KB, 64 KB, 1 MB, 4 MB, more
struct SizeHist {
    uint64_t n[kSizeBuckets] = {}, bytes[kSizeBuckets] = {};
};
SizeHist g_hist[kUploadKinds];
struct PageStats {
    uint64_t first = 0, dup = 0, same = 0, changed = 0;
};
PageStats g_pages[kUploadKinds];
struct PageSeen {
    uint64_t hash = 0, submit = 0;
};
std::unordered_map<uintptr_t, PageSeen> g_page_seen;
int size_bucket(uint64_t s) { return s <= 4096 ? 0 : s <= (64u << 10) ? 1 : s <= (1u << 20) ? 2 : s <= (4u << 20) ? 3 : 4; }
struct VertexSource {
    uint64_t bytes = 0, copies = 0, size = 0;
};
std::unordered_map<const uint8_t*, VertexSource> g_vertex_by_addr;

bool upload_stats_on() {
    static const bool on = std::getenv("NX_UPLOAD_STATS") != nullptr;
    return on;
}

void count_copy(const uint8_t* p, uint64_t size) {
    KindStats& k = g_kind[g_upload_kind < kUploadKinds ? g_upload_kind : kUploadOther];
    k.copies++, k.bytes += size;
    const uint64_t h = hash_memory(p, size);
    auto& last = g_last_hash[p];
    if (last.first == size && last.second == h) k.same += size;
    last = {size, h};
    if (g_upload_kind == kUploadVertex) {  // which vertex buffers the bytes come from (top list in dedup_log)
        auto& v = g_vertex_by_addr[p];
        v.bytes += size, v.copies++, v.size = size;
    }
    const int kind = g_upload_kind < kUploadKinds ? g_upload_kind : kUploadOther;
    const int b = size_bucket(size);
    g_hist[kind].n[b]++, g_hist[kind].bytes[b] += size;
    const uintptr_t end = (uintptr_t(p) + size) & ~uintptr_t(4095);
    for (uintptr_t a = (uintptr_t(p) + 4095) & ~uintptr_t(4095); a < end; a += 4096) {
        const uint64_t ph = hash_memory(reinterpret_cast<const uint8_t*>(a), 4096);
        PageSeen& ps = g_page_seen[a];
        if (ps.submit == g_generation) g_pages[kind].dup++;
        else if (ps.submit == 0) g_pages[kind].first++;
        else if (ps.hash == ph) g_pages[kind].same++;
        else g_pages[kind].changed++;
        ps.hash = ph, ps.submit = g_generation;
    }
}

}  // namespace

UploadKind g_upload_kind = kUploadOther;
uint64_t g_upload_min_offset = 0;

Upload dedup_upload(const uint8_t* p, uint64_t size, uint64_t padded, uint64_t align) {
    if (!p || !size) {
        Upload u = upload_alloc(padded, align);
        std::memset(u.ptr, 0, padded);
        return u;
    }
    frame_guest_read(p, size, kReadDedup);  // deferred write-backs over these bytes land first (nvn_frames.cpp)
    static const bool off = std::getenv("NX_NO_DEDUP") != nullptr;  // bisecting: a fresh copy every time
    Front& f = front_of(p);
    const Entry* e = nullptr;
    if (!off && f.generation == g_generation && f.p == p) {
        e = &f.entry;  // the map's entry for p
    } else if (CopySlot* slot = off ? nullptr : copy_find(p)) {
        f = {p, g_generation, slot->entry};
        e = &f.entry;
    }
    if (e && e->size == size && e->padded == padded && e->upload.offset % align == 0) {
        g_saved += padded;
        return e->upload;
    }
    Upload u;
    if ((g_upload_kind == kUploadVertex || g_upload_kind == kUploadIndex) &&
        page_cache_upload(p, size, padded, align, u)) {  // nvn_page_cache.cpp: a mirror, nothing copied into the ring
        copy_store(p, {size, padded, u});
        f = {p, g_generation, {size, padded, u}};
        return u;
    }
    if (g_upload_kind != kUploadOther && cache_upload(p, size, padded, align, u)) {  // nvn_buffer_cache.cpp
        copy_store(p, {size, padded, u});
        f = {p, g_generation, {size, padded, u}};
        return u;
    }
    u = upload_alloc(padded, align);
    std::memcpy(u.ptr, p, size);
    if (padded > size) std::memset(u.ptr + size, 0, padded - size);
    if (upload_stats_on()) count_copy(p, size);
    copy_store(p, {size, padded, u});
    f = {p, g_generation, {size, padded, u}};
    g_copied += padded;
    g_submit_kind_bytes[g_upload_kind < kUploadKinds ? g_upload_kind : kUploadOther] += padded;
    return u;
}

void dedup_reset() {
    ++g_generation;
    if (!g_generation) {
        for (CopySlot& slot : g_copies) slot.generation = 0;
        g_generation = 1;
    }
    g_copy_count = 0;
    index_range_reset();
    for (uint64_t& b : g_submit_kind_bytes) b = 0;
}

// "vertex 120 MB, index 5 MB, ...": what this submit has copied into the ring so far (not the textures or the
// written blocks, which take other paths: the difference to the ring's own count is theirs).
void dedup_submit_breakdown(char* out, size_t n) {
    static const char* names[kUploadKinds] = {"other", "vertex", "index", "uniform/storage"};
    size_t at = 0;
    if (n) out[0] = 0;
    for (int k = 0; k < kUploadKinds && at + 1 < n; k++) {
        const int w = std::snprintf(out + at, n - at, "%s%s %.0f MB", at ? ", " : "", names[k],
                                    double(g_submit_kind_bytes[k]) / 1e6);
        if (w < 0) break;
        at += size_t(w);
    }
}

void dedup_invalidate(const uint8_t* p, uint64_t size) {
    page_cache_invalidate(p, size);  // the page hashes of this submit
    // Invalidation is much rarer than upload lookup. Scan the flat table so overlapping ranges are still exact.
    const uintptr_t lo = uintptr_t(p), hi = lo + size;
    for (CopySlot& slot : g_copies) {
        if (slot.generation != g_generation) continue;
        const uintptr_t start = uintptr_t(slot.p), end = start + slot.entry.size;
        if (start < hi && end > lo) {
            if (Front& f = front_of(slot.p); f.p == slot.p) f.generation = 0;
            slot.generation = 0;
            --g_copy_count;
        }
    }
}

// c[0], the NVN driver's constants, as far as shaders read them (docs/nvn/TRANSLATOR.md): buffer descriptors
// {u64 GPU address; u32 size} of what is bound, which compute shaders search to find the buffer an address falls in
// (they wrote garbage with a zero table), and the maximum point size.
std::array<uint8_t, 0x1000> driver_constants(ShaderStage stage, int st) {
    std::array<uint8_t, 0x1000> c0{};
    auto put = [&](uint32_t at, const Range& r) {
        if (at + 12 > c0.size()) return;
        std::memcpy(c0.data() + at, &r.address, 8);
        uint32_t size = uint32_t(std::min<uint64_t>(r.size, 0xFFFFFFFFu));
        std::memcpy(c0.data() + at + 8, &size, 4);
    };
    if (stage == ShaderStage::Compute) {
        for (int n = 0; n < 16; n++) put(0x210 + 0x10 * n, g_state.uniforms[st][n]);
        for (int n = 0; n < 16; n++) put(0x310 + 0x10 * n, g_state.storage[st][n]);
    } else {
        int hw = stage == ShaderStage::Fragment ? 4 : 0;  // Maxwell stage numbers: vertex 0, fragment 4
        for (int n = 0; n < 16; n++) put(0x110 + 0x100 * hw + 0x10 * n, g_state.storage[st][n]);
        float max_point = 64.0f;
        std::memcpy(c0.data() + 0x610, &max_point, 4);
    }
    return c0;
}

void dedup_log() {
    if (g_copied + g_saved)
        log("nvn: uploads copied %.1f MB, reused %.1f MB", double(g_copied) / 1e6, double(g_saved) / 1e6);
    g_copied = g_saved = 0;
    static const char* names[kUploadKinds] = {"other", "vertex", "index", "uniform/storage"};
    for (int k = 0; k < kUploadKinds; k++) {
        KindStats& s = g_kind[k];
        if (s.copies)
            log("nvn: uploads of %s: %llu copies, %.1f MB, %.0f%% the same as their last copy", names[k],
                (unsigned long long)s.copies, double(s.bytes) / 1e6, 100.0 * double(s.same) / double(s.bytes));
        s = {};
    }
    for (int k = 0; k < kUploadKinds; k++) {
        const SizeHist& h = g_hist[k];
        uint64_t total = 0;
        for (int b = 0; b < kSizeBuckets; b++) total += h.n[b];
        if (total)
            log("nvn: copy sizes of %s: <=4K %llu (%.1f MB), <=64K %llu (%.1f MB), <=1M %llu (%.1f MB), <=4M %llu (%.1f MB), "
                "more %llu (%.1f MB)", names[k], (unsigned long long)h.n[0], double(h.bytes[0]) / 1e6,
                (unsigned long long)h.n[1], double(h.bytes[1]) / 1e6, (unsigned long long)h.n[2], double(h.bytes[2]) / 1e6,
                (unsigned long long)h.n[3], double(h.bytes[3]) / 1e6, (unsigned long long)h.n[4], double(h.bytes[4]) / 1e6);
        const PageStats& ps = g_pages[k];
        const uint64_t pages = ps.first + ps.dup + ps.same + ps.changed;
        if (pages)
            log("nvn: 4 KB pages of %s copies: %.1f MB; %.0f%% repeated inside the submit, %.0f%% the same as the last submit "
                "that copied them, %.0f%% changed, %.0f%% first seen", names[k], double(pages) * 4096.0 / 1e6,
                100.0 * double(ps.dup) / double(pages), 100.0 * double(ps.same) / double(pages),
                100.0 * double(ps.changed) / double(pages), 100.0 * double(ps.first) / double(pages));
        g_hist[k] = {}, g_pages[k] = {};
    }
    if (g_page_seen.size() > (1u << 20)) g_page_seen.clear();  // 4 GB of pages: start over
    if (!g_vertex_by_addr.empty()) {
        std::vector<std::pair<const uint8_t*, VertexSource>> top(g_vertex_by_addr.begin(), g_vertex_by_addr.end());
        std::sort(top.begin(), top.end(), [](const auto& x, const auto& y) { return x.second.bytes > y.second.bytes; });
        for (size_t k = 0; k < top.size() && k < 8; k++)
            log("nvn: vertex source #%zu at %llx: %.1f MB in %llu copies (last %llu bytes)", k,
                (unsigned long long)uintptr_t(top[k].first), double(top[k].second.bytes) / 1e6,
                (unsigned long long)top[k].second.copies, (unsigned long long)top[k].second.size);
        log("nvn: vertex sources: %zu addresses", top.size());
        g_vertex_by_addr.clear();
    }
}

}  // namespace nx::nvn::replay
