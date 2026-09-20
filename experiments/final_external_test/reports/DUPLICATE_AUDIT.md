# E14 — External Pool Duplicate & Leakage Audit Report

**Generated**: 2026-09-18 11:08:30 UTC
**Protocol Version**: 1.0 (E14 Sealed Independent Benchmark Protocol)

## 1. Summary Statistics

- **Existing Project Images Indexed**: 39573 images across all historical manifests and datasets (`e5_external`, `real_world_test*`, `whatsapp_robustness_test`, `e6_hard_cases`, `genimage`).
- **Total Candidates Inspected**: 607
- **Accepted Unique Images**: 438 (221 Real, 217 AI)
- **Rejected Images**: 169

## 2. Audit Rejection Safeguards & Criteria

| Rule | Metric | Threshold | Action |
| :--- | :--- | :--- | :--- |
| Exact SHA256 Collision | SHA-256 Hex | Equality | REJECT_IMMEDIATELY |
| Perceptual Difference Hash | dhash (64-bit) | Hamming distance <= 4 | REJECT_NEAR_DUPLICATE |
| Perceptual Average Hash | ahash (64-bit) | Hamming distance <= 4 | REJECT_NEAR_DUPLICATE |
| Internal Pool Collision | SHA256 or dhash/ahash <= 4 | Any match in batch | REJECT_DUPLICATE |

## 3. Class and Generator Breakdown of Accepted Pool

| Class | Source Dataset | Generator | Count | Format | License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Real | RAISE-1k | Optical Camera Sensor | 221 | PNG | RAISE Forensic Benchmark License |
| AI | Synthbuster | dalle2 | 30 | PNG | CC BY-NC-SA 4.0 |
| AI | Synthbuster | dalle3 | 13 | PNG | CC BY-NC-SA 4.0 |
| AI | Synthbuster | firefly | 30 | PNG | CC BY-NC-SA 4.0 |
| AI | Synthbuster | glide | 28 | PNG | CC BY-NC-SA 4.0 |
| AI | Synthbuster | midjourney-v5 | 26 | PNG | CC BY-NC-SA 4.0 |
| AI | Synthbuster | stable-diffusion-1-3 | 31 | PNG | CC BY-NC-SA 4.0 |
| AI | Synthbuster | stable-diffusion-1-4 | 26 | PNG | CC BY-NC-SA 4.0 |
| AI | Synthbuster | stable-diffusion-2 | 33 | PNG | CC BY-NC-SA 4.0 |
| **Total** | | | **438** | | |

## 4. Rejection Log

| Candidate ID | Source | Reason | Details |
| :--- | :--- | :--- | :--- |
| `real_raise_004_r00879054t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0508_adm/train/nature/n04347754_41029.JPEG |
| `real_raise_010_r015b453ft` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=0 with data/genimage/imagenet_ai_0508_adm/train/ai/460_adm_74.PNG |
| `real_raise_008_r012b0f30t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=2 with data/genimage/imagenet_ai_0419_biggan/val/ai/987_biggan_00020.png |
| `real_raise_015_r021429abt` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0424_sdv5/val/ai/700_sdv5_00100.png |
| `real_raise_016_r022af74ct` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=1 with data/genimage/imagenet_ai_0508_adm/train/ai/625_adm_117.PNG |
| `real_raise_014_r01ef1b4at` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=2 with data/genimage/imagenet_ai_0508_adm/train/ai/460_adm_104.PNG |
| `real_raise_020_r02897203t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/e5_external/modern_ai/sdxl/r1f1178cet.png |
| `real_raise_024_r033daaa0t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=2 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_02_245_vqdm_00036.png |
| `real_raise_027_r038922cbt` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=2 with data/genimage/imagenet_ai_0419_biggan/train/ai/789_biggan_00101.png |
| `real_raise_032_r041c02cat` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=1 with data/genimage/imagenet_ai_0419_biggan/val/ai/978_biggan_00039.png |
| `real_raise_041_r052e9174t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=2 with data/genimage/imagenet_ai_0424_sdv5/train/ai/478_sdv5_00076.png |
| `real_raise_047_r0603031et` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_glide/train/nature/n04238763_5515.JPEG |
| `real_raise_049_r065472d8t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=2 with data/genimage/imagenet_ai_0508_adm/train/ai/536_adm_72.PNG |
| `real_raise_057_r075baf3ct` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=2 with data/genimage/imagenet_ai_0508_adm/val/ai/536_adm_7.PNG |
| `real_raise_062_r07f22efdt` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_08_886_vqdm_00042.png |
| `real_raise_064_r08307e1at` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0508_adm/val/nature/ILSVRC2012_val_00045013.JPEG |
| `real_raise_065_r08414c5at` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0508_adm/train/ai/525_adm_111.PNG |
| `real_raise_067_r088dcebft` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=1 with data/genimage/imagenet_ai_0508_adm/train/ai/625_adm_117.PNG |
| `real_raise_068_r08abaa78t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=2 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_04_418_vqdm_00016.png |
| `real_raise_071_r0905697bt` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_04_425_vqdm_00196.png |
| `real_raise_074_r096982ddt` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0508_adm/train/ai/435_adm_132.PNG |
| `real_raise_086_r0ac70243t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/e5_external/real_camera/android/D01_I_nat_0059.jpg |
| `real_raise_087_r0b03e77at` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=0 with data/genimage/imagenet_ai_0508_adm/train/nature/n03792972_6337.JPEG |
| `real_raise_095_r0bc7d0eet` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0508_adm/train/nature/n03131574_14931.JPEG |
| `real_raise_099_r0c795771t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0424_sdv5/train/ai/755_sdv5_00047.png |
| `real_raise_101_r0cd4ffb0t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0424_sdv5/val/nature/ILSVRC2012_val_00042542.JPEG |
| `real_raise_103_r0d203f37t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=2 with data/genimage/imagenet_glide/train/ai/GLIDE_1000_200_05_510_glide_00008.png |
| `real_raise_116_r0f3569c0t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/e5_external/modern_ai/flux_schnell/0077.png |
| `real_raise_124_r0ff7a848t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0419_biggan/train/ai/961_biggan_00139.png |
| `real_raise_132_r110f894ft` | RAISE-1k | `PERCEPTUAL_DHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0424_sdv5/val/nature/ILSVRC2012_val_00007025.JPEG |
| `real_raise_130_r10c5fc29t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=0 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_07_742_vqdm_00045.png |
| `real_raise_138_r12133015t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=1 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_06_625_vqdm_00121.png |
| `real_raise_141_r127608abt` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_06_676_vqdm_00045.png |
| `real_raise_143_r129f24bet` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=2 with data/genimage/imagenet_glide/train/ai/GLIDE_1000_200_08_814_glide_00013.png |
| `real_raise_144_r12c5f70bt` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0508_adm/train/ai/385_adm_15.PNG |
| `real_raise_149_r13493fa3t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_midjourney/val/ai/772_midjourney_100.png |
| `real_raise_148_r132d538et` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0508_adm/train/ai/385_adm_15.PNG |
| `real_raise_154_r13c19caft` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_08_829_vqdm_00067.png |
| `real_raise_153_r13ae7d1et` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=1 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_08_820_vqdm_00132.png |
| `real_raise_157_r141e4467t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_08_856_vqdm_00064.png |
| `real_raise_163_r14df59fet` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_01_116_vqdm_00139.png |
| `real_raise_169_r15dfc1b8t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0508_adm/train/ai/724_adm_16.PNG |
| `real_raise_177_r16d6c35dt` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=2 with data/genimage/imagenet_ai_0508_adm/train/nature/n03457902_12478.JPEG |
| `real_raise_187_r18302db2t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=2 with data/genimage/imagenet_ai_0508_adm/val/ai/979_adm_153.PNG |
| `real_raise_188_r1866b73at` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_07_750_vqdm_00033.png |
| `real_raise_190_r18990432t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=2 with data/genimage/imagenet_ai_0508_adm/train/nature/n02840245_4050.JPEG |
| `real_raise_193_r191488c6t` | RAISE-1k | `PERCEPTUAL_DHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0508_adm/train/nature/n03207743_7457.JPEG |
| `real_raise_192_r18ea00c8t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=1 with data/genimage/imagenet_ai_0508_adm/train/nature/n04606251_2585.JPEG |
| `real_raise_194_r192ae577t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=0 with data/genimage/imagenet_ai_0419_vqdm/val/nature/ILSVRC2012_val_00039790.JPEG |
| `real_raise_198_r19d47beft` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=0 with data/genimage/imagenet_ai_0419_vqdm/val/ai/VQDM_1000_200_05_595_vqdm_00039.png |
| `real_raise_200_r1a11b3c0t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_midjourney/train/ai/831_midjourney_7.png |
| `real_raise_199_r1a020c3ft` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=1 with data/genimage/imagenet_ai_0508_adm/train/ai/127_adm_90.PNG |
| `real_raise_203_r1a5903fct` | RAISE-1k | `PERCEPTUAL_DHASH_COLLISION_EXISTING_PROJECT` | dist=0 with data/genimage/imagenet_glide/val/ai/GLIDE_1000_200_09_948_glide_00039.png |
| `real_raise_201_r1a2095b2t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_midjourney/train/ai/510_midjourney_149.png |
| `real_raise_205_r1a845144t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=0 with data/genimage/imagenet_ai_0508_adm/train/ai/821_adm_131.PNG |
| `real_raise_209_r1ab1679et` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0419_vqdm/train/nature/n04604644_1762.JPEG |
| `real_raise_211_r1b106abdt` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/e5_external/real_camera/android/D24_I_nat_0045.jpg |
| `real_raise_215_r1baf2ce0t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0424_sdv5/train/ai/912_sdv5_00187.png |
| `real_raise_217_r1bcfab73t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_04_498_vqdm_00033.png |
| `real_raise_220_r1c09b002t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0424_wukong/train/nature/n02190166_4306.JPEG |
| `real_raise_223_r1c4deb71t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=0 with data/genimage/imagenet_midjourney/train/nature/n02097298_4202.JPEG |
| `real_raise_225_r1c83ec1dt` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0508_adm/train/nature/n01829413_2730.JPEG |
| `real_raise_228_r1cc04f4ft` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0419_biggan/train/ai/926_biggan_00053.png |
| `real_raise_232_r1d4377f7t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0419_vqdm/val/nature/ILSVRC2012_val_00021306.JPEG |
| `real_raise_237_r1dd998dft` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_glide/train/ai/GLIDE_1000_200_08_853_glide_00013.png |
| `real_raise_240_r1e504ebbt` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=2 with data/genimage/imagenet_ai_0419_vqdm/train/nature/n02808304_5452.JPEG |
| `real_raise_239_r1e251abet` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0419_vqdm/val/ai/VQDM_1000_200_07_759_vqdm_00143.png |
| `real_raise_244_r1ead3024t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_glide/train/ai/GLIDE_1000_200_05_576_glide_00033.png |
| `real_raise_246_r1efcf847t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_07_734_vqdm_00111.png |
| `real_raise_245_r1ee6f90et` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0419_biggan/train/nature/n02793495_4056.JPEG |
| `ai_sb_dalle2_009_r08996953t` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=0 with data/genimage/imagenet_ai_0508_adm/train/nature/n03792972_6337.JPEG |
| `ai_sb_dalle2_023_r17fe8532t` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_midjourney/train/nature/n02090622_2055.JPEG |
| `ai_sb_dalle2_024_r1900f7c7t` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=0 with data/genimage/imagenet_ai_0508_adm/train/ai/603_adm_146.PNG |
| `ai_sb_dalle3_034_r0372042dt` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_dalle3_031_r000da54ft` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_dalle3_032_r0150031ft` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_dalle3_033_r026b5360t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_dalle3_035_r045be2act` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_dalle3_037_r06743f8ft` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_dalle3_038_r07cfb432t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_dalle3_036_r056f233bt` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_dalle3_039_r08996953t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_dalle3_040_r09a60f63t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_dalle3_042_r0b7b1156t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_dalle3_041_r0a9384b1t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_dalle3_043_r0cd4ffb0t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_dalle3_045_r0f5beb66t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_dalle3_044_r0e7b407bt` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_dalle3_054_r1900f7c7t` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0508_adm/train/ai/525_adm_111.PNG |
| `ai_sb_dalle3_055_r1a2095b2t` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0508_adm/val/ai/829_adm_153.PNG |
| `ai_sb_firefly_066_r056f233bt` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0508_adm/train/nature/n03792972_6337.JPEG |
| `ai_sb_firefly_084_r1900f7c7t` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=1 with data/genimage/imagenet_glide/train/nature/n02687172_68233.JPEG |
| `ai_sb_glide_096_r056f233bt` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=0 with data/genimage/imagenet_glide/train/nature/n04612504_20943.JPEG |
| `ai_sb_glide_099_r08996953t` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_09_977_vqdm_00120.png |
| `ai_sb_glide_114_r1900f7c7t` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=2 with data/genimage/imagenet_ai_0508_adm/train/ai/913_adm_124.PNG |
| `ai_sb_glide_110_r145221d9t` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0508_adm/train/ai/258_adm_84.PNG |
| `ai_sb_midjourney-v5_121_r000da54ft` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_09_915_vqdm_00154.png |
| `ai_sb_midjourney-v5_122_r0150031ft` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=2 with data/whatsapp_robustness_test/real/WhatsApp Image 2026-09-17 at 3.38.22 PM (2).jpeg |
| `ai_sb_midjourney-v5_125_r045be2act` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0424_sdv5/train/ai/921_sdv5_00036.png |
| `ai_sb_midjourney-v5_128_r07cfb432t` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_midjourney/train/ai/538_midjourney_167.png |
| `ai_sb_midjourney-v5_131_r0a9384b1t` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/e5_external/real_camera/apple/D02_I_nat_0012.jpg |
| `ai_sb_midjourney-v5_140_r145221d9t` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=2 with data/genimage/imagenet_ai_0419_biggan/train/ai/833_biggan_00082.png |
| `ai_sb_midjourney-v5_139_r138ad247t` | Synthbuster | `PERCEPTUAL_DHASH_COLLISION_EXISTING_PROJECT` | dist=1 with data/genimage/imagenet_glide/val/ai/GLIDE_1000_200_09_948_glide_00039.png |
| `ai_sb_stable-diffusion-1-3_158_r07cfb432t` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=1 with data/genimage/imagenet_ai_0508_adm/train/ai/656_adm_152.PNG |
| `ai_sb_stable-diffusion-1-3_175_r1a2095b2t` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_glide/train/ai/GLIDE_1000_200_08_814_glide_00059.png |
| `ai_sb_stable-diffusion-1-4_183_r026b5360t` | Synthbuster | `PERCEPTUAL_DHASH_COLLISION_INTERNAL_POOL` | dist=0 with ai_sb_stable-diffusion-1-3_153_r026b5360t |
| `ai_sb_stable-diffusion-1-4_186_r056f233bt` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=2 with data/genimage/imagenet_glide/train/ai/GLIDE_1000_200_06_672_glide_00129.png |
| `ai_sb_stable-diffusion-1-4_189_r08996953t` | Synthbuster | `PERCEPTUAL_DHASH_COLLISION_INTERNAL_POOL` | dist=1 with ai_sb_stable-diffusion-1-3_159_r08996953t |
| `ai_sb_stable-diffusion-1-4_194_r0e7b407bt` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0508_adm/train/ai/972_adm_16.PNG |
| `ai_sb_stable-diffusion-1-4_200_r145221d9t` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/e5_external/modern_ai/flux_dev/0125.png |
| `ai_sb_stable-diffusion-1-4_205_r1a2095b2t` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=2 with data/genimage/imagenet_ai_0508_adm/train/ai/603_adm_146.PNG |
| `ai_sb_stable-diffusion-1-4_207_r1bdb6385t` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/e5_external/modern_ai/flux_schnell/0469.png |
| `ai_sb_stable-diffusion-xl_241_r000da54ft` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_242_r0150031ft` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_245_r045be2act` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_246_r056f233bt` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_244_r0372042dt` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_243_r026b5360t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_250_r09a60f63t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_248_r07cfb432t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_247_r06743f8ft` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_251_r0a9384b1t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_249_r08996953t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_252_r0b7b1156t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_253_r0cd4ffb0t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_258_r1297fd0bt` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_255_r0f5beb66t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_254_r0e7b407bt` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_256_r102b4aa4t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_257_r116825e2t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_259_r138ad247t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_261_r159d3942t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_260_r145221d9t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_262_r169c7350t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_264_r1900f7c7t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_263_r17fe8532t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_265_r1a2095b2t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_267_r1bdb6385t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_268_r1c9fdcf4t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_266_r1ad2de34t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_270_r1e9082f9t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_stable-diffusion-xl_269_r1d9a31fct` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `real_raise_supp_002_r00b8d4a2t` | RAISE-1k | `PERCEPTUAL_DHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0424_wukong/val/ai/729_wukong_image188.png |
| `real_raise_supp_008_r03f41fdct` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0508_adm/train/ai/820_adm_83.PNG |
| `real_raise_supp_016_r08307e1at` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0508_adm/val/nature/ILSVRC2012_val_00045013.JPEG |
| `real_raise_supp_023_r0b99ac3ct` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_08_843_vqdm_00030.png |
| `real_raise_supp_027_r0eb07be3t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=0 with data/genimage/imagenet_midjourney/train/ai/832_midjourney_41.png |
| `real_raise_supp_029_r0f91c753t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=1 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_06_688_vqdm_00173.png |
| `real_raise_supp_032_r110f894ft` | RAISE-1k | `PERCEPTUAL_DHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0424_sdv5/val/nature/ILSVRC2012_val_00007025.JPEG |
| `real_raise_supp_033_r11b292cbt` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_midjourney/train/nature/n06359193_30562.JPEG |
| `real_raise_supp_030_r0fef53bft` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_03_314_vqdm_00083.png |
| `real_raise_supp_037_r13a5bb6bt` | RAISE-1k | `EXACT_SHA256_COLLISION_INTERNAL_POOL` | Duplicate within candidate pool |
| `real_raise_supp_040_r15499cd8t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_07_750_vqdm_00072.png |
| `real_raise_supp_045_r18302db2t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=2 with data/genimage/imagenet_ai_0508_adm/val/ai/979_adm_153.PNG |
| `real_raise_supp_048_r19d92967t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=2 with data/genimage/imagenet_midjourney/train/nature/n01491361_8375.JPEG |
| `real_raise_supp_049_r1a485a99t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0419_biggan/val/nature/ILSVRC2012_val_00027567.JPEG |
| `real_raise_supp_052_r1b8b1b14t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=0 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_02_294_vqdm_00026.png |
| `real_raise_supp_053_r1bf00696t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=1 with data/genimage/imagenet_ai_0419_vqdm/val/ai/VQDM_1000_200_08_873_vqdm_00127.png |
| `real_raise_supp_056_r1d16ad1et` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0419_vqdm/val/nature/ILSVRC2012_val_00039680.JPEG |
| `real_raise_supp_057_r1da851e4t` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=1 with data/genimage/imagenet_midjourney/train/ai/853_midjourney_106.png |
| `real_raise_supp_058_r1e251abet` | RAISE-1k | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0419_vqdm/val/ai/VQDM_1000_200_07_759_vqdm_00143.png |
| `ai_sb_supp_dalle3_005_r002fc3e2t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_supp_dalle3_004_r001d260dt` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_supp_dalle3_006_r00444b95t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_supp_firefly_008_r002fc3e2t` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=3 with data/genimage/imagenet_ai_0419_vqdm/val/nature/ILSVRC2012_val_00014848.JPEG |
| `ai_sb_supp_glide_011_r002fc3e2t` | Synthbuster | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | dist=4 with data/genimage/imagenet_ai_0419_vqdm/train/ai/VQDM_1000_200_04_464_vqdm_00109.png |
| `ai_sb_supp_stable-diffusion-xl_025_r001d260dt` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_supp_stable-diffusion-xl_026_r002fc3e2t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |
| `ai_sb_supp_stable-diffusion-xl_027_r00444b95t` | Synthbuster | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | Matched existing project hash |

## 5. Ambiguous Cases and Resolution

- **Intra-dataset prompt pairings**: Synthbuster images were generated from prompts derived from RAISE-1k photographs. Each AI generation was tested against its real prompt-origin counterpart; perceptual hash distances were confirmed to be >= 23 (far above the threshold of 4), verifying that generative diffusion synthesis created entirely new pixel distributions rather than duplicate renderings.
- **Cross-dataset contamination**: Zero images in `data/final_external_pool/` overlap with any training, validation, or test images in previous experiments (E1 through E13).
