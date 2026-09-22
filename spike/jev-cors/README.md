# Spike: can an extension call Jev with no backend?

**Answered: yes. Confirmed 2026-09-22.** An MV3 extension with `host_permissions`
for `https://api.typesafe.ai/*` calls the API directly. No proxy, no tunnel, no
server, and no third party handling anyone's API key.

## The measurement

A minimal extension page fetched `POST /v1/systemone` with a deliberately bogus key.
The question was never whether it would authenticate -- it was whether the request
would leave the browser at all:

```
{"stage":"alive"}
{"stage":"responded","status":401,"ms":123, ...authentication_error... }
```

It returned **401 in 123ms** rather than throwing. The request reached the server, so
nothing blocked it client-side.

## Why this was in doubt

Measured against the live API beforehand:

- A **preflight** from an unlisted origin is answered `400 Disallowed CORS origin`.
  That is what stops a plain web page: a POST carrying JSON and an `Authorization`
  header is not a simple request, so the browser preflights it and the call never
  leaves. A static demo page therefore cannot reach Jev -- and should not, since that
  would put a key in public JavaScript.
- A **POST** carrying `Origin: chrome-extension://...` and a bad key returned `401`,
  not the CORS `400`. The origin filter gates preflights only, not real requests.

Chrome exempts fetches to hosts in `host_permissions` from CORS, so no preflight is
sent. That exemption was the one part taken on trust, and the run above confirms it.

## Running it yourself

`popup.html` is the interactive version: load unpacked, paste a real key, press the
button, and it makes one real two-question call.

```
chrome://extensions -> Developer mode -> Load unpacked -> this directory
```

| Result | Meaning |
|---|---|
| **200** with two answers | Works end to end, key and all. |
| **401** | CORS fine, key wrong. Still proves the architecture. |
| `fetch threw` / "Failed to fetch" | Chrome blocked it. A proxy would be required. |
| **400** mentioning CORS | The origin allowlist rejected it. A proxy would be required. |

## Note on automating this

**Branded Chrome cannot do it.** Chrome 137 removed `--load-extension`, and Chrome 142
removed the `--disable-features=DisableLoadExtensionCommandLineSwitch` workaround too.
On Chrome 153 the flag is silently ignored: the extension never loads, nothing executes,
and no error is logged -- which reads exactly like a broken script.

Use **Chrome for Testing**, which still supports the flag:

```bash
npx @puppeteer/browsers install chrome@stable
```

Pin the extension id by putting the base64 DER public key in `manifest.json` as `key`,
then navigate straight to `chrome-extension://<id>/<page>.html`. An MV3 service worker
stays asleep under automation, so drive an extension *page*, not the worker. Do not use
a content script: those are subject to CORS and would answer a different question.
