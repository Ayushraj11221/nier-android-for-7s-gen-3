#!/usr/bin/env bash
# NieR:Automata NCE APK on a Linux host (GitHub Actions): the same steps as android/build_apk.ps1 -Game nier -Nce
# (cmake NDK -> libnier.so + libnierimport.so + the adrenotools hooks + the shader translator libspirvharness.so ->
# aapt2 -> javac/d8 for games/nier/java -> zipalign -> apksigner with the debug key).
# Needs: ANDROID_HOME with NDK 28.2.13676358, build-tools 35.0.0, platforms android-35; cmake, ninja, a JDK (javac),
# zip; Vulkan headers in $VULKAN_SDK/Include (capital I, as CMakeLists.txt asks); the harness already built in
# private/build/spirv_harness_android_nier/libspirvharness.so (games/nier/tools/build_harness_android.sh).
# Output: private/nier/apk_nce/nier.apk
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
sdk="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"
[ -n "$sdk" ] || { echo "ANDROID_HOME is not set" >&2; exit 1; }
ndk="$sdk/ndk/28.2.13676358"
bt="$sdk/build-tools/35.0.0"
jar="$sdk/platforms/android-35/android.jar"
for p in "$ndk/build/cmake/android.toolchain.cmake" "$bt/aapt2" "$jar"; do
    [ -e "$p" ] || { echo "missing $p" >&2; exit 1; }
done
[ -f "${VULKAN_SDK:-/nonexistent}/Include/vulkan/vulkan.h" ] || { echo "VULKAN_SDK/Include/vulkan/vulkan.h not found" >&2; exit 1; }
harness="$root/private/build/spirv_harness_android_nier/libspirvharness.so"
[ -f "$harness" ] || { echo "missing $harness (games/nier/tools/build_harness_android.sh)" >&2; exit 1; }

priv="$root/private/nier"
out="$priv/apk_nce"
build="$priv/ba_nce"
llvm="$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin"

if [ ! -f "$build/build.ninja" ]; then
    cmake -S "$root" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 -DANDROID_STL=c++_static -DNX_GAME=nier -DNX_NCE=ON
fi
cmake --build "$build" --target nier nierimport

rm -rf "$out"
mkdir -p "$out/lib/arm64-v8a"
put_lib() { "$llvm/llvm-strip" --strip-unneeded -o "$out/lib/arm64-v8a/$(basename "$1")" "$1"; }
put_lib "$build/libnier.so"
put_lib "$build/libnierimport.so"
put_lib "$harness"
for h in libhook_impl.so libmain_hook.so libfile_redirect_hook.so libgsl_alloc_hook.so; do
    put_lib "$build/adrenotools/src/hook/$h"
done

# resources, with the project's own launcher icon when there is one
"$bt/aapt2" compile --dir "$root/android/res" -o "$out/res.zip"
# A copy of the manifest with versionCode = 1000 + the CI run number: always rising, so each build installs as an
# update (versionName stays the one in games/nier/AndroidManifest.xml).
manifest="$out/AndroidManifest.xml"
vcode=$((1000 + ${GITHUB_RUN_NUMBER:-0}))
sed "s/android:versionCode=\"[0-9]*\"/android:versionCode=\"$vcode\"/" "$root/games/nier/AndroidManifest.xml" > "$manifest"
grep -q "android:versionCode=\"$vcode\"" "$manifest" || { echo "could not set versionCode in the manifest" >&2; exit 1; }
echo "versionCode $vcode"
reszips=("$out/res.zip")
if [ -d "$root/games/nier/icon/res" ]; then
    "$bt/aapt2" compile --dir "$root/games/nier/icon/res" -o "$out/icon.zip"
    reszips+=("$out/icon.zip")
    sed -i 's/<application/<application android:icon="@mipmap\/ic_launcher"/' "$manifest"
fi
"$bt/aapt2" link -o "$out/base.apk" -I "$jar" --manifest "$manifest" "${reszips[@]}"

# the native libraries (compressed, as the PowerShell build does)
(cd "$out" && zip -q -r -9 base.apk lib)

# the launcher's Java (games/nier/java): javac against android.jar, d8 to classes.dex
mkdir -p "$out/classes" "$out/dex"
find "$root/games/nier/java" -name '*.java' > "$out/sources.txt"
javac -encoding UTF-8 -source 1.8 -target 1.8 -Xlint:-options -bootclasspath "$jar" -d "$out/classes" @"$out/sources.txt"
find "$out/classes" -name '*.class' > "$out/classes.txt"
"$bt/d8" --release --min-api 29 --lib "$jar" --output "$out/dex" @"$out/classes.txt"
zip -q -j "$out/base.apk" "$out/dex/classes.dex"

"$bt/zipalign" -f -p 4 "$out/base.apk" "$out/aligned.apk"
release_ks="$HOME/.android/nier-release.keystore"
if [ -f "$release_ks" ] && [ -n "${NIER_KS_PASS:-}" ]; then
    # the fixed key: passwords come from the environment, never the command line
    export NIER_KEY_PASS="${NIER_KEY_PASS:-$NIER_KS_PASS}"
    "$bt/apksigner" sign --ks "$release_ks" --ks-pass env:NIER_KS_PASS --ks-key-alias "${NIER_KEY_ALIAS:-nierkey}" \
        --key-pass env:NIER_KEY_PASS --out "$out/nier.apk" "$out/aligned.apk"
    echo "signed with the fixed release key"
else
    "$bt/apksigner" sign --ks "$HOME/.android/debug.keystore" --ks-pass pass:android --key-pass pass:android \
        --out "$out/nier.apk" "$out/aligned.apk"
    echo "signed with a throwaway debug key: this APK will not install over an older build"
fi
"$bt/apksigner" verify --print-certs "$out/nier.apk"
echo "APK: $out/nier.apk ($(du -h "$out/nier.apk" | cut -f1))"
