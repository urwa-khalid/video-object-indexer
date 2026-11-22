# Write-up

## 1. Your timings (100 words)
Final run is 12,590 s (about 3 h 30 min, 3.5 h) for 1,650 videos and 128.27 h of video. I only ran a final version with some brief test that took around 90 mins and helped me building up the choices I took. The slow every-frame reference ran at about 3x real time (about 40 h projected, an estimate). The biggest saver was sampling 4 frames per second instead of every frame. That gave about 9x on the 12-video test: 1,131 s to 127 s, with 7x fewer frames through the model.

## 2. Your pipeline (150 words)
Read the Parquet mp4 bytes and queue them.
Two workers, each with its own model on its own GPU.
PyAV decode, skipping non-reference frames.
Keep 4 frames per second.
Letterbox, then fp16 batches of 16.
NMS with confidence 0.4 and IoU 0.7, counting only the 5 classes.
Per second: an object counts if it appears in at least half the frames, and people is the median.
Write one JSON line per second, with an error line if a video fails.



## 3. Where the time goes (150 words)

Inference takes 70.8%, decode 27.0%, and preprocess 1.7%. It was measured with --profile, which puts perf_counter timers around each stage in each worker.

## 4. Checking correctness (100 words)

I compared against an every-frame, official-predict, fp32 reference on 24 videos using evaluate_results.py. At 4 fps, micro F1 was about 0.98, people exact about 0.95, and within-one about 0.99, all above the 0.95 / 0.85 / 0.95 thresholds. You also ran the 25 unit tests. Macro F1 is below 0.95 at 4 fps, and the README mention it.

## 5. Working with AI (100 words)

AI told me that I should start running on the whole batch and then wait for results. Which is quite time taking and I have to wait while run finished. So, instead I build a small sample run to test different apporaches and if they passed the thresholds, only then I ran the only final and full run, therefore I am not claimimg to have different first and final versions.

## 6. Next steps (100 words)

Tune workers, batch size and decode threads, which stayed at defaults.
Use all 4 vCPUs and overlap decode with inference.
Try 3 fps, which also passes the three thresholds.
Check accuracy on more videos, especially the rare classes (dog, laptop, phone).