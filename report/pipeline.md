# Gesture detection pipeline

## Dataset creation

### Pre processing

Convert MTS file to AVI using:

```bash
$ ffmpeg -i 00000.MTS -b 18000k -ac 2 -ab 320k -deinterlace -s 1920x1080 00000.AVI
```

Using dyads trajectory ((x,y,t) coordinates relative to video time and an horizontal plane) to find appropriate time stamps when dyad enters and leaves the part of the path that can be processed by the camera.

Extract interesting portion (between previously extracted time stamps) using:

```bash
$ ffmpeg -ss 00:xx:xx -i 00000.AVI  -t 00:00:yy -vcodec copy -acodec copy dyad_gesture_1.avi
```

Crop video around dyad using:

```bash
$ ffmpeg -i dyad_gesture_1.avi -filter:v "crop=in_w/2:in_h/2:in_w/3:in_h/2" -c:a copy cropped.avi
```
