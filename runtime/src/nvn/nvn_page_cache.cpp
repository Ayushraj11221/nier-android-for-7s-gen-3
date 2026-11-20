// Page mirror of guest vertex and index data, kept across submits (NieR; the P5R buffer cache of nvn_buffer_cache.cpp
// keys on an exact address + size + hash and skips ranges over 4 MB, and NieR's trimmed vertex ranges change size from
// draw to draw, so it never got a hit worth having).
//
// Here the unit is a 4 KB page of guest memory, not a range. Guest memory is cut into windows of 512 KB on a 256 KB grid:
// any range of up to 256 KB lies inside one window (the first run on the phone: ranges averaged 62 KB, and 4 MB windows
// ran the arena out: 144 windows touched in 300 frames against 32 slots). A window has up to three mirrors (host-visible Vulkan buffers of the
// window's size, carved from one arena buffer and laid out like guest memory), each with the hash of every page it holds. A draw that needs [p, p+size)
// hashes the guest pages it covers (CRC32 instructions, once per page per submit) and binds a mirror whose pages all
// match: no byte is copied into the upload ring. If none matches, the mirror the GPU is done with that differs least is
// brought up to date (only the pages that differ are copied); if every mirror is still in flight the range goes to the
// ring as before. A mirror is never written while a submit that read it is recording or running, hence several.
//
// A page no mirror has copied yet is never read by a submit, so it can be filled into a mirror at any time, even one a
// running submit reads; only a page whose bytes changed needs an idle mirror. (The first version filled new pages into
// whichever mirror was idle and spread a window over all three, then found them all in flight.)
//
// Pages that change nearly every time they are asked for (effect vertex buffers the CPU rewrites each frame) mark their
// window "streaming" for a while: it goes to the ring. The cache also times itself and switches itself off when hashing
// and copying cost more than the ring copies it saved.
//
// Slots of the arena are taken from its top: a vertex copy is bound at (offset - first vertex's byte), which must not
// be negative (draw_inputs keeps the copy's place in the stream), so it has to lie at an arena offset of at least that.
// NX_PAGE_CACHE=1: on (off by default). NX_PAGE_CACHE_MB: memory for mirrors (default 128). NX_PAGE_CACHE_VERIFY=1: every
// hit compared with the guest bytes (slow, and logs "a hit differs" if it served stale data).
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "nvn_draw_state.h"
#include "platform.h"

#if defined(__aarch64__) && defined(__clang__) && defined(__linux__)  // Android: CRC32 instructions, found by getauxval
#include <arm_acle.h>
#include <sys/auxv.h>
#define NX_PAGE_CRC 1
#endif

namespace nx::nvn {
uint64_t hash_memory(const uint8_t* p, uint64_t size);  // nvn_sampler.cpp
uint64_t frame_recording_serial();                     // nvn_frames.cpp
}  // namespace nx::nvn

namespace nx::nvn::replay {

using namespace nx::vk;

namespace {

constexpr uint64_t kPage = 4096;
constexpr uint64_t kGrid = 256ull << 10;               // windows start on this grid ...
constexpr uint64_t kWindow = 512ull << 10;              // ... and are this long: a range of up to kGrid fits in one
constexpr uint32_t kPages = uint32_t(kWindow / kPage);  // 128
constexpr uint64_t kMinRange = 16 << 10;               // smaller ranges: the ring copy is cheaper than a page's hash
constexpr uint32_t kVersions = 3;                      // mirrors per window (frames in flight + the one recording)
constexpr uint64_t kEvictAfter = 30;                    // a window asked for more recently keeps its mirrors when the arena is full
constexpr uint64_t kStaleSubmits = 600;                // a window nobody asked for in this long is freed
constexpr uint64_t kStreamingSubmits = 240;            // a streaming window stays in the ring this long
constexpr uint64_t kJudgeSubmits = 300;                // the self-check runs this often
constexpr uint64_t kOffSubmits = 1800;                 // and switches the cache off for this long when it loses
constexpr uint64_t kJudgeMinBytes = 32ull << 20;          // the self-check needs this much served to judge
constexpr double kRingNsPerByte = 0.30;                // what a ring copy costs (0.34-0.42 ms per MB measured)

struct Mirror {
    int32_t slot = -1;      // the arena slot (kWindow bytes at slot * kWindow); -1: no mirror
    uint8_t* ptr = nullptr;  // the slot's bytes
    uint64_t last_use = 0;  // the serial of the last submit that bound it (0: never)
    std::vector<uint64_t> hash;  // kPages entries: the hash of each page it holds; 0: never copied (a real hash has bit 0 set)
    bool exists() const { return slot >= 0; }
};

struct Window {
    Mirror mirror[kVersions];
    uint64_t guest_hash[kPages];  // hash of each guest page, valid when its bit in hashed_pages is set
    uint64_t hash_generation = 0;
    uint64_t hashed_pages[2] = {};
    uint64_t touched = 0;          // g_submits when last asked
    uint64_t streaming_until = 0;  // g_submits
    uint64_t asked_pages = 0, changed_pages = 0;  // since the last judgement of this window
};

std::unordered_map<uint64_t, Window> g_windows;  // by guest address / kGrid
uint64_t g_submits = 0;
uint64_t g_gen = 1;  // guest page hashes of this generation are valid (one per submit)
uint64_t g_mirror_bytes = 0, g_budget = 0;
// The arena: one host-visible buffer of budget() bytes, made at the first use.
struct Arena {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint8_t* ptr = nullptr;
    uint32_t slots = 0;
    std::vector<uint32_t> free_slots;
    bool failed = false;
} g_arena;
uint64_t g_off_until = 0;
uint64_t g_judged_at = 0, g_judge_ns = 0, g_judge_bytes = 0;

struct {
    uint64_t hits = 0, hit_bytes = 0;        // ranges served by a mirror with nothing copied
    uint64_t updates = 0;                    // ranges served after copying some pages
    uint64_t cold_pages = 0, changed_pages = 0, copied_bytes = 0;
    uint64_t hash_ns = 0, copy_ns = 0, hashed_pages = 0;
    uint64_t in_flight = 0, streaming = 0, no_memory = 0, wrong = 0;
    uint64_t small = 0, small_bytes = 0, big = 0, big_bytes = 0;  // ranges outside [kMinRange, kGrid]: the ring
} g_pc_stats;
uint64_t g_full_at = 0;  // g_submits when the arena had no free slot and nothing to free

bool on() {
    static const bool v = [] {
        const char* e = std::getenv("NX_PAGE_CACHE");
        return e && e[0] == '1';
    }();
    return v;
}

bool verify_on() {
    static const bool v = std::getenv("NX_PAGE_CACHE_VERIFY") != nullptr;
    return v;
}

uint64_t page_bits(uint32_t first, uint32_t end, uint32_t word) {
    const uint32_t lo = std::max(first, word * 64), hi = std::min(end, word * 64 + 64);
    if (lo >= hi) return 0;
    const uint32_t count = hi - lo, shift = lo - word * 64;
    return count == 64 ? UINT64_MAX : ((uint64_t(1) << count) - 1) << shift;
}

uint64_t budget() {
    if (!g_budget) {
        long mb = std::getenv("NX_PAGE_CACHE_MB") ? std::atol(std::getenv("NX_PAGE_CACHE_MB")) : 128;
        g_budget = uint64_t(std::clamp(mb, 16l, 1024l)) << 20;
    }
    return g_budget;
}

// ---- the hash of a 4 KB page -------------------------------------------------------------------------------------
#ifdef NX_PAGE_CRC
// Four independent CRC32C streams (the instruction has a latency of a few cycles, so four in flight keep it busy),
// folded into 64 bits. Always has bit 0 set (0 means "no page").
__attribute__((target("crc"))) uint64_t page_hash_crc(const uint8_t* p) {
    const uint64_t* q = reinterpret_cast<const uint64_t*>(p);
    uint32_t c0 = 0x11111111u, c1 = 0x22222222u, c2 = 0x33333333u, c3 = 0x44444444u;
    for (int i = 0; i < 128; i += 8) {
        __builtin_prefetch(q + i + 64), __builtin_prefetch(q + 128 + i + 64);  // 512 bytes ahead in each stream
        __builtin_prefetch(q + 256 + i + 64), __builtin_prefetch(q + 384 + i + 64);
        for (int k = 0; k < 8; k++) {
            c0 = __crc32cd(c0, q[i + k]);
            c1 = __crc32cd(c1, q[128 + i + k]);
            c2 = __crc32cd(c2, q[256 + i + k]);
            c3 = __crc32cd(c3, q[384 + i + k]);
        }
    }
    const uint64_t a = (uint64_t(c0) << 32) | c1, b = (uint64_t(c2) << 32) | c3;
    return (a ^ (b * 0x9E3779B97F4A7C15ull)) | 1;
}
bool crc_available() {
    static const bool v = (getauxval(AT_HWCAP) & (1ul << 7)) != 0;  // HWCAP_CRC32
    return v;
}
#endif

uint64_t page_hash(const uint8_t* p) {
#ifdef NX_PAGE_CRC
    if (crc_available()) return page_hash_crc(p);
#endif
    return nvn::hash_memory(p, kPage) | 1;
}

// ---- mirrors -----------------------------------------------------------------------------------------------------
bool ensure_arena() {
    if (g_arena.buffer) return true;
    if (g_arena.failed) return false;
    g_arena.failed = true;  // until it works
    VkDevice d = vk_device();
    const uint64_t bytes = (budget() / kWindow) * kWindow;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = kUploadUsage;
    VkBuffer buffer = VK_NULL_HANDLE;
    if (p_vkCreateBuffer(d, &bi, nullptr, &buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    p_vkGetBufferMemoryRequirements(d, buffer, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = upload_memory_type(req.memoryTypeBits);  // as the upload ring
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* p = nullptr;
    if (p_vkAllocateMemory(d, &ai, nullptr, &memory) != VK_SUCCESS || p_vkBindBufferMemory(d, buffer, memory, 0) ||
        p_vkMapMemory(d, memory, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS) {
        p_vkDestroyBuffer(d, buffer, nullptr);
        if (memory) p_vkFreeMemory(d, memory, nullptr);
        log("nvn: page cache: no memory for the %llu MB arena: off", (unsigned long long)(bytes >> 20));
        return false;
    }
    g_arena.buffer = buffer, g_arena.memory = memory, g_arena.ptr = static_cast<uint8_t*>(p);
    g_arena.slots = uint32_t(bytes / kWindow);
    for (uint32_t k = 0; k < g_arena.slots; k++) g_arena.free_slots.push_back(k);
    g_arena.failed = false;
    log("nvn: page cache: arena of %llu MB (%u slots of %llu KB)", (unsigned long long)(bytes >> 20), g_arena.slots,
        (unsigned long long)(kWindow >> 10));
    return true;
}

// The free slot highest in the arena whose copy of the page offset `at` lies at or above `min_off`; false when none.
bool take_slot(Mirror& m, uint64_t at, uint64_t min_off) {
    size_t best = SIZE_MAX;
    for (size_t k = 0; k < g_arena.free_slots.size(); k++)
        if (g_arena.free_slots[k] * kWindow + at >= min_off &&
            (best == SIZE_MAX || g_arena.free_slots[k] > g_arena.free_slots[best]))
            best = k;
    if (best == SIZE_MAX) return false;
    m.slot = int32_t(g_arena.free_slots[best]);
    m.ptr = g_arena.ptr + uint64_t(m.slot) * kWindow;
    m.last_use = 0;
    m.hash.assign(kPages, 0);
    g_arena.free_slots.erase(g_arena.free_slots.begin() + best);
    g_mirror_bytes += kWindow;
    return true;
}

void destroy_mirror(Mirror& m) {
    if (!m.exists()) return;
    g_arena.free_slots.push_back(uint32_t(m.slot));
    g_mirror_bytes -= kWindow;
    m = Mirror{};
}

// No submit that recorded or ran a command reading the mirror is still recording or running.
bool mirror_idle(const Mirror& m, uint64_t rec, uint64_t done) { return m.last_use < rec && m.last_use <= done; }

uint64_t completed_serial() { return async_gpu() ? frame_completed() : UINT64_MAX; }  // synchronous: all waited for

// One mirror freed, from the window nobody asked for the longest (not `keep`); false when none is idle.
bool make_room(const Window* keep, uint64_t rec, uint64_t done) {
    Window* best = nullptr;
    for (auto& kv : g_windows) {
        Window& w = kv.second;
        if (&w == keep || w.touched + kEvictAfter > g_submits) continue;  // not one asked for lately: that would thrash
        bool idle = false;
        for (Mirror& m : w.mirror) idle |= m.exists() && mirror_idle(m, rec, done);
        if (idle && (!best || w.touched < best->touched)) best = &w;
    }
    if (!best) return false;
    Mirror* oldest = nullptr;
    for (Mirror& m : best->mirror)
        if (m.exists() && mirror_idle(m, rec, done) && (!oldest || m.last_use < oldest->last_use)) oldest = &m;
    destroy_mirror(*oldest);
    return true;
}

// Copies the pages [a0, a1) of the window that the mirror does not hold (hash differs) from guest memory; runs of
// pages in one memcpy. Returns the pages copied; `changed` of them held other bytes before (the rest were empty).
uint32_t update_mirror(Window& w, Mirror& m, const uint8_t* base, uint32_t a0, uint32_t a1, uint32_t& changed) {
    const uint64_t t0 = plat::monotonic_ns();
    uint32_t copied = 0;
    changed = 0;
    for (uint32_t i = a0; i < a1;) {
        if (m.hash[i] == w.guest_hash[i]) {
            i++;
            continue;
        }
        uint32_t j = i;
        while (j < a1 && m.hash[j] != w.guest_hash[j]) {
            if (m.hash[j]) changed++;
            m.hash[j] = w.guest_hash[j];
            j++;
        }
        std::memcpy(m.ptr + uint64_t(i) * kPage, base + uint64_t(i) * kPage, uint64_t(j - i) * kPage);
        copied += j - i;
        i = j;
    }
    g_pc_stats.copy_ns += plat::monotonic_ns() - t0;
    return copied;
}

// Pages [a0, a1) the mirror holds with other bytes (`changed`) and pages it has never copied (`cold`).
void count_mismatch(const Window& w, const Mirror& m, uint32_t a0, uint32_t a1, uint32_t& changed, uint32_t& cold) {
    changed = cold = 0;
    for (uint32_t i = a0; i < a1; i++) {
        if (m.hash[i] == w.guest_hash[i]) continue;
        (m.hash[i] ? changed : cold)++;
    }
}

void judge() {
    if (g_submits < g_judged_at + kJudgeSubmits) return;
    g_judged_at = g_submits;
    if (g_judge_bytes < kJudgeMinBytes) return;  // too little served to tell (a menu): keep adding up
    // What the mirrors cost (hashing and copying pages) against what the ring copies of the bytes they served would have.
    const double saved = double(g_judge_bytes) * kRingNsPerByte;
    if (g_judge_bytes && double(g_judge_ns) > saved) {
        g_off_until = g_submits + kOffSubmits;
        log("nvn: page cache: costs more than it saves (%.1f ms for %.1f MB served, a ring copy of those is %.1f ms): "
            "off for %llu submits", double(g_judge_ns) / 1e6, double(g_judge_bytes) / 1e6, saved / 1e6,
            (unsigned long long)kOffSubmits);
    }
    g_judge_ns = g_judge_bytes = 0;
}

}  // namespace

void page_cache_begin_submit() {
    if (!on()) return;
    g_submits++;
    g_gen++;  // the guest pages are hashed again: the game wrote its memory since
    judge();
    if (g_submits % 64) return;
    const uint64_t rec = frame_recording_serial(), done = completed_serial();
    for (auto it = g_windows.begin(); it != g_windows.end();) {
        Window& w = it->second;
        bool any = false;
        for (Mirror& m : w.mirror) {
            if (m.exists() && w.touched + kStaleSubmits < g_submits && mirror_idle(m, rec, done)) destroy_mirror(m);
            any |= m.exists();
        }
        it = (!any && w.touched + kStaleSubmits < g_submits) ? g_windows.erase(it) : std::next(it);
    }
}

// Guest bytes written during the replay: the pages' hashes of this submit are stale.
void page_cache_invalidate(const uint8_t* p, uint64_t size) {
    if (!on() || !size || g_windows.empty()) return;
    const uint64_t first = uintptr_t(p) / kGrid, last = (uintptr_t(p) + size - 1) / kGrid;
    for (uint64_t g = first >= 1 ? first - 1 : 0; g <= last; g++) {  // a window starting a grid step back reaches in
        auto it = g_windows.find(g);
        if (it == g_windows.end()) continue;
        const uintptr_t base = uintptr_t(g * kGrid);
        const uintptr_t lo = std::max<uintptr_t>(uintptr_t(p), base), hi = std::min<uintptr_t>(uintptr_t(p) + size, base + kWindow);
        if (lo >= hi) continue;
        Window& w = it->second;
        if (w.hash_generation != g_gen) continue;
        const uint32_t a0 = uint32_t((lo - base) / kPage), a1 = uint32_t((hi - 1 - base) / kPage + 1);
        w.hashed_pages[0] &= ~page_bits(a0, a1, 0);
        w.hashed_pages[1] &= ~page_bits(a0, a1, 1);
    }
}

bool page_cache_upload(const uint8_t* p, uint64_t size, uint64_t padded, uint64_t align, Upload& out) {
    if (!on() || !p || g_submits == 0 || g_submits < g_off_until) return false;
    if (size < kMinRange) {
        g_pc_stats.small++, g_pc_stats.small_bytes += size;
        return false;
    }
    if (size > kGrid) {
        g_pc_stats.big++, g_pc_stats.big_bytes += size;
        return false;
    }
    if (uintptr_t(p) % (align ? align : 1) != 0 || padded < size) return false;  // the offset in the mirror keeps p's
    const uint64_t min_off = g_upload_min_offset;  // the caller binds at (offset - min_off): not below zero
    const uint64_t win = uintptr_t(p) / kGrid;
    Window& w = g_windows[win];
    w.touched = g_submits;
    if (w.streaming_until > g_submits) {
        g_pc_stats.streaming++;
        return false;
    }
    const uint8_t* base = reinterpret_cast<const uint8_t*>(uintptr_t(win * kGrid));
    const uint64_t at = uintptr_t(p) - uintptr_t(base);  // < kGrid, and at + size <= kWindow
    if (at + padded > kWindow) return false;  // the zero padding after the bytes would not fit (rare)
    const uint32_t a0 = uint32_t(at / kPage), a1 = uint32_t((at + size - 1) / kPage + 1);
    const uint64_t rec = frame_recording_serial(), done = completed_serial();

    // No mirror of this window and the arena is full this submit: nothing can be served, so no hashing either.
    bool has_mirror = false;
    for (const Mirror& m : w.mirror) has_mirror |= m.exists();
    if (!has_mirror && g_full_at == g_submits) {
        g_pc_stats.no_memory++;
        return false;
    }

    // A two-word bitmap makes a repeated/overlapping same-submit range skip the per-page generation checks entirely.
    if (w.hash_generation != g_gen) {
        w.hash_generation = g_gen;
        w.hashed_pages[0] = w.hashed_pages[1] = 0;
    }
    const uint64_t needed0 = page_bits(a0, a1, 0), needed1 = page_bits(a0, a1, 1);
    const uint64_t missing0 = needed0 & ~w.hashed_pages[0], missing1 = needed1 & ~w.hashed_pages[1];
    if (missing0 || missing1) {
        const uint64_t t0 = plat::monotonic_ns();
        for (uint32_t i = a0; i < a1; i++) {
            const uint32_t word = i >> 6;
            const uint64_t bit = uint64_t(1) << (i & 63);
            if (w.hashed_pages[word] & bit) continue;
            w.guest_hash[i] = page_hash(base + uint64_t(i) * kPage);
            w.hashed_pages[word] |= bit;
            g_pc_stats.hashed_pages++;
        }
        const uint64_t hash_ns = plat::monotonic_ns() - t0;
        g_pc_stats.hash_ns += hash_ns;
        g_judge_ns += hash_ns;  // counted whether or not the range ends up in a mirror: wasted hashing is a cost
    }

    // 1. A mirror holding no changed page: bound as it is, never-used pages filled into it first. Never-used pages are
    //    read by no submit, so this is safe while the mirror is in flight. The fewest pages to fill wins.
    // 2. Else the idle mirror that differs least: changed pages are overwritten, so no submit may still read it.
    Mirror* chosen = nullptr;
    uint32_t chosen_changed = 0, chosen_cold = 0, best_cold = UINT32_MAX, idle_cost = UINT32_MAX;
    Mirror* idle = nullptr;
    uint32_t idle_changed = 0, idle_cold = 0;
    for (Mirror& m : w.mirror) {
        if (!m.exists() || uint64_t(m.slot) * kWindow + at < min_off) continue;
        uint32_t ch, co;
        count_mismatch(w, m, a0, a1, ch, co);
        if (ch == 0) {
            if (co < best_cold) best_cold = co, chosen = &m, chosen_changed = 0, chosen_cold = co;
        } else if (mirror_idle(m, rec, done) && ch + co < idle_cost) {
            idle_cost = ch + co, idle = &m, idle_changed = ch, idle_cold = co;
        }
    }
    if (!chosen && idle) chosen = idle, chosen_changed = idle_changed, chosen_cold = idle_cold;
    if (!chosen) {
        // 3. Else a new mirror (room made by freeing an idle one of a window not asked for lately).
        Mirror* fresh = nullptr;
        for (Mirror& m : w.mirror)
            if (!m.exists()) {
                fresh = &m;
                break;
            }
        if (!fresh) {
            g_pc_stats.in_flight++;  // three mirrors, each holding changed pages and read by submits not done: the ring
            return false;
        }
        if (!ensure_arena()) {
            g_pc_stats.no_memory++;
            return false;
        }
        for (int tries = 0; !take_slot(*fresh, at, min_off); tries++)
            if (tries == 3 || !make_room(&w, rec, done)) {
                if (g_arena.free_slots.empty()) g_full_at = g_submits;
                g_pc_stats.no_memory++;
                return false;
            }
        chosen = fresh, chosen_changed = 0, chosen_cold = a1 - a0;
    }

    uint64_t copy_ns = 0;
    if (chosen_changed + chosen_cold) {
        uint32_t changed = 0;
        const uint64_t c0 = g_pc_stats.copy_ns;
        const uint32_t copied = update_mirror(w, *chosen, base, a0, a1, changed);
        copy_ns = g_pc_stats.copy_ns - c0;
        g_pc_stats.updates++;
        g_pc_stats.copied_bytes += uint64_t(copied) * kPage;
        g_pc_stats.changed_pages += changed;
        g_pc_stats.cold_pages += copied - changed;
        w.changed_pages += changed;
    } else {
        g_pc_stats.hits++, g_pc_stats.hit_bytes += size;
        if (verify_on() && std::memcmp(chosen->ptr + at, p, size) != 0) {
            static int said = 0;
            if (said++ < 20)
                log("nvn: page cache: a hit differs from the guest bytes (%llu bytes at %p)", (unsigned long long)size,
                    (const void*)p);
            g_pc_stats.wrong++;
        }
    }
    if (verify_on() && chosen_changed + chosen_cold && std::memcmp(chosen->ptr + at, p, size) != 0) {
        static int said = 0;
        if (said++ < 20)
            log("nvn: page cache: an updated mirror differs from the guest bytes (%llu bytes at %p)",
                (unsigned long long)size, (const void*)p);
        g_pc_stats.wrong++;
    }
    // A window whose pages keep changing between asks (CPU-rewritten vertices) goes to the ring for a while.
    w.asked_pages += a1 - a0;
    if (w.asked_pages >= 512) {
        if (w.changed_pages * 2 > w.asked_pages) w.streaming_until = g_submits + kStreamingSubmits;
        w.asked_pages = w.changed_pages = 0;
    }
    chosen->last_use = rec;
    g_judge_ns += copy_ns;
    g_judge_bytes += size;
    out = {g_arena.buffer, uint64_t(chosen->slot) * kWindow + at, chosen->ptr + at};
    return true;
}

void page_cache_log() {
    auto& s = g_pc_stats;
    if (s.hits + s.updates + s.in_flight + s.streaming + s.no_memory + s.small + s.big) {
        size_t mirrors = 0;
        for (auto& kv : g_windows)
            for (const Mirror& m : kv.second.mirror) mirrors += m.exists();
        log("nvn: page cache: %llu ranges with nothing copied (%.1f MB), %llu ranges updated (%.1f MB of pages copied: "
            "%llu pages changed, %llu new); hashing %llu pages %.1f ms, copying %.1f ms; to the ring: %llu in flight, "
            "%llu streaming, %llu no memory; to the ring by size: %llu under %llu KB (%.1f MB), %llu over %llu KB (%.1f MB); "
            "%zu windows, %zu mirrors, %.0f of %.0f MB%s%s",
            (unsigned long long)s.hits, double(s.hit_bytes) / 1e6, (unsigned long long)s.updates,
            double(s.copied_bytes) / 1e6, (unsigned long long)s.changed_pages, (unsigned long long)s.cold_pages,
            (unsigned long long)s.hashed_pages, double(s.hash_ns) / 1e6, double(s.copy_ns) / 1e6,
            (unsigned long long)s.in_flight, (unsigned long long)s.streaming, (unsigned long long)s.no_memory,
            (unsigned long long)s.small, (unsigned long long)(kMinRange >> 10), double(s.small_bytes) / 1e6,
            (unsigned long long)s.big, (unsigned long long)(kGrid >> 10), double(s.big_bytes) / 1e6, g_windows.size(), mirrors, double(g_mirror_bytes) / 1e6, double(budget()) / 1e6,
            g_submits < g_off_until ? "; OFF (cost more than it saved)" : "",
            s.wrong ? "; HITS THAT DIFFER (NX_PAGE_CACHE_VERIFY)" : "");
    }
    s = {};
}

}  // namespace nx::nvn::replay
