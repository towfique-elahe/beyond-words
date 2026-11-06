# Beyond Words web front end

A static page that talks to the Beyond Words API on Modal. No build step.

## Files

- `index.html`, `style.css`, `app.js`: the whole page.
- `app.js` line 3: `API_BASE`, the Modal deployment (no trailing slash).

## Put it on cPanel

1. In cPanel, open **File Manager**, go to `public_html/`, create a folder
   (for example `dialect`) and upload the three files into it.
2. Open `https://your-domain/dialect/`. The page pings `/api/health` on load,
   which also wakes the model server (about 30 s the first time after idle).
3. Microphone recording needs HTTPS; cPanel's AutoSSL certificate is enough.

For a subdomain, point the subdomain's document root at the same folder.

## What the page does

- Records with `MediaRecorder` (webm/opus on Chrome and Firefox, m4a on
  Safari and iPhone) with a level meter and a 20 s cap, or accepts an upload.
- Sends the audio to `POST /api/audio` (optional `text` and `transcribe`),
  or a typed sentence to `POST /api/text`, with a 150 s timeout and two
  retries so a cold start does not fail the request.
- Shows probability bars for the sound route, the words route and their
  fusion, the transcript, and the words outside standard Bangla with the
  region each word points to.

## API

```
GET  /api/health
POST /api/text   {"text": "কিতা করতায়"}
POST /api/audio  multipart: file=<audio>, text=<optional>, transcribe=1|0
```
