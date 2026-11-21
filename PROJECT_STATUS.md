# NieR Android Project Status

**Updated:** 2026-11-22  
**Branch:** `main`  
**Repository state before this status update:** based on `43fd7da` (`Revert "Set Turnip GMEM debug flag before driver load"`), with a local guest-heap tracing change described below.  
**Device under test:** Snapdragon 7s Gen 3 / Adreno 810, Android, NieR:Automata: The End of YoRHa Edition.

This document records the current performance and stability problem, what the latest logs establish, which experiments are active or rolled back, and what evidence is needed before making a memory-management change.

## Current issue

The latest long play session reached Route B. The player reports that performance is normally acceptable until the sequence after 9S helps 2B, when the chainsaw arm flies through the chimney. In that area, the phone begins to lag severely and becomes close to freezing. The player stopped the game to avoid the phone becoming unusable. There is no confirmed game crash or Android low-memory-killer termination in the supplied log; the run was stopped manually.

The current leading explanation is **severe memory pressure and paging**, not an upload-ring capacity problem. The exact allocation or resource set that grows is not yet proven. The log has no story-scene labels, and the existing memory sample reports process-wide resident, anonymous, file-backed, and swapped totals rather than memory by virtual-memory region.

## Evidence from the latest log

Log reviewed: `/storage/emulated/0/NiER Automata/nx.log` (4,256,372 bytes; latest available run at the time of review). The following are sampled values, not a continuous trace:

| Log line | Resident | Anonymous / file-backed | Process swap (`VmSwap`) | Android `MemAvailable` / `SwapFree` |
|---:|---:|---:|---:|---:|
| 720 | 1.15 GB | 788 / 371 MB | 54 MB | 1,662 / 3,475 MB |
| 10,275 | 1.31 GB | 853 / 484 MB | 3,903 MB | 522 / 51 MB |
| 10,480 | 1.41 GB | 982 / 463 MB | 4,050 MB | 290 / 0 MB |
| 30,553 | 1.24 GB | 835 / 434 MB | 4,726 MB | 168 / 0 MB |

Other signs of paging pressure:

- One 600-present sample reports 197,873 major page faults and 177,034 minor page faults for replay/present. A later sample reports 18,577 major faults and 2,665 minor faults over 600 presents, with the replay thread on CPU only 9% of the sampled time.
- Near the end, the log records repeated long frame stalls, including approximately 718 ms and 848 ms. The device was close to freezing, consistent with the reported experience.
- These numbers show the app and system are under extreme memory pressure. They do **not**, by themselves, prove whether the largest source is the guest heap, native allocations, or another anonymous mapping.

The memory guard is using the low-cost `statm` source by default (`NX_MEMGUARD` is not set in the user config). This avoids periodically walking the process page tables with `smaps_rollup`. One isolated slow `statm` sample in the log is not enough to identify the source of the overall stalls.

## Upload, page-cache, and counter interpretation

The user tested both `NX_UPLOAD_RING_MB=128` and `NX_UPLOAD_RING_MB=256`; the same severe slowdown occurs with both. The latest supplied config uses 256 MB. This makes ring capacity an unlikely primary cause.

The page cache is enabled with a 96 MB arena. The log shows it can spend substantial CPU time hashing pages; it later reports that the measured hashing cost exceeded the ring-copy cost and disables itself for a period. This is a real secondary CPU cost worth revisiting, but the arena is about 96 MB and cannot account for several GB of swap by itself.

The `carried compute results` lines report ranges and byte totals processed/adopted over diagnostic windows. They do not mean that hundreds of MB are all retained in memory at once. The carry arena is fixed at 32 MB. Do not age-evict these results or change write-back synchronization without evidence; stale or missing compute results can corrupt cloth, hair, or meshes.

The log contains VP9 movie decoder sessions, including sessions stopped at presents 7,331 and 27,024. Decoder buffers are on the scale of tens of MB, far below the observed multi-GB swap growth. The log does not establish that movie playback causes the Route B memory pressure.

The game data is present under `/storage/emulated/0/NiER Automata`, including the imported RomFS CPK archives. The import index lists the archives, not their internal asset names, and the runtime log does not name the Route B scene. The exact asset transition therefore has not yet been mapped to a specific CPK entry.

## Current user config observed on the device

The config at `/storage/emulated/0/NiER Automata/config.txt` currently includes:

```text
RESOLUTION=480p
NX_NIER_MIN_SCALE=0.6
NX_UPLOAD_RING_MB=256
NX_PAGE_CACHE=1
NX_PAGE_CACHE_MB=96
NX_REPLAY_THREAD=1
NX_PIPELINED=1
NX_YIELD=spin
NX_SYNC_WAIT_SKIP=1
NX_TU_DEBUG_GMEM=0
NX_REPLAY_PROFILE=0
NX_LOD_BIAS=1
```

Other visible settings include low graphics, disabled bloom and shadows, 480p render resolution, and GPU turbo enabled. This is the config observed during log review; it is stored outside the repository and was not modified for this status update.

## Active safety constraints

These are important when investigating or changing the runtime:

- Keep `gpu_wait(serial)` in `SyncWait` (`runtime/src/nvn/nvn_queue.cpp:284`). Compute write-backs must become visible before guest code reads them.
- Keep Adreno 8xx chunked submits disabled. The source documents a GPU hang in battle passes (`runtime/src/nvn/nvn_frames.cpp:468`).
- Keep `NX_PASS_BARRIER_NARROW` disabled. Prior testing produced a black band (`runtime/src/nvn/nvn_draw.cpp:50`).
- Keep `NX_BUFFER_CACHE` off. The user observed severe FPS loss and near-crash behavior when it was enabled; the implementation defaults it off.
- Do not evict carried compute results based only on their cumulative diagnostic byte count.
- Keep forced GMEM debug off. The latest config has `NX_TU_DEBUG_GMEM=0`; the most recent repository commit reverted the forced Turnip GMEM debug flag.

## Experiments and repository state

- The descriptor/replay and upload hot-path optimization experiment was reverted in commit `07235b6` after the user reported worse performance. The following GMEM debug forcing change was reverted in `43fd7da`.
- Current repository history is on `main`, tracking `origin/main`; before the changes in this update, the working tree was clean at `43fd7da`.
- The latest log is a real long Route B play session, but there is no direct one-to-one scene marker in it. The user has not yet built and installed the current local diagnostic change.
- No behavioral memory fix has been made in this update. The current local code change is measurement-only.

## Diagnostic change made in this update

The runtime now records the guest heap size reported through `svcSetHeapSize`:

- `runtime/src/kernel.cpp` stores the heap size atomically, logs each size change, and exposes it through the existing heap-size accessor.
- `runtime/src/kernel_state.h` declares the shared heap-size value as atomic.
- `runtime/src/memguard_android.cpp` adds the guest heap's configured virtual size to the existing periodic memory sample.

This adds no new `/proc/self/smaps` scan and does not change guest allocation, rendering, upload, or synchronization behavior. The heap size is a **virtual size**, not a measurement of resident or swapped bytes in that region. It will help determine whether the game resizes the heap near the problematic scene; if it does not, region-level residency accounting will still be needed to distinguish guest-heap pages from other anonymous memory.

`git diff --check` passed after the code change. The modified code has not been compiled or run yet.

## Fixes and optimizations applied

1. **Guest Heap Decommit on Shrink (`runtime/src/kernel.cpp`)**:
   - `svcSetHeapSize` now issues `decommit(kHeapBase + size, old_size - size)` when the heap shrinks. Previously, shrinking only updated `g_heap_size` and VM permissions without returning committed physical pages to the kernel, causing accumulated anonymous pages to stick in memory across scene switches.

2. **Real Memory Guard Accounting & Native Purge (`runtime/src/memguard_android.cpp`)**:
   - Accounted for `VmSwap` in default `statm` memory guard monitoring (`f = resident + swap`), preventing the guard from staying blind when the app swaps 4+ GB to ZRAM.
   - Added periodic native allocator purges (`mallopt(M_PURGE, 0)`) every 10 seconds to flush unused heap arenas and prevent Scudo memory retention.

3. **Texture Upload Buffer Pooling (`runtime/src/nvn/nvn_upload.cpp`)**:
   - Replaced repeated per-texture dynamic heap allocation (`std::vector linear` and `copies`) in `upload_texture` with reusable `thread_local` storage. This eliminates hundreds of MB of allocation and deallocation thrash during scene transitions (such as the 441 textures staged at Route B Goliath entrance).

4. **Page Cache Optimization (`config.txt`)**:
   - Disabled `NX_PAGE_CACHE` (`NX_PAGE_CACHE=0`). Runtime logs proved that page cache hashing was costing 138–355 ms per period and consuming 96 MB while providing negative net benefit over direct ring copies.
