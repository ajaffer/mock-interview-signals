// Does a Chrome extension get to call api.typesafe.ai directly?
//
// A web page cannot: the API runs an origin allowlist, and a preflight from an
// unlisted origin is answered `400 Disallowed CORS origin`. A POST carrying
// JSON and an Authorization header is not a simple request, so the browser
// preflights it and the call never leaves.
//
// An extension should not hit that, for two independent reasons measured
// before this was written:
//   1. Chrome exempts fetches to hosts in `host_permissions` from CORS, so no
//      preflight is sent at all.
//   2. Even with `Origin: chrome-extension://...` set explicitly, the POST
//      handler answered 401 rather than the CORS 400 -- it reached auth, so
//      the origin filter does not gate real requests.
//
// This runs the whole path -- CORS, auth, response shape -- against the live
// API, because the first reason is the one still taken on trust.

const ENDPOINT = "https://api.typesafe.ai/v1/systemone";

const out = document.getElementById("out");
const keyInput = document.getElementById("key");

// Convenience only. A real build would ask whether to persist the key at all.
chrome.storage.local.get("key").then(({ key }) => {
  if (key) keyInput.value = key;
});

function show(html) {
  out.innerHTML = html;
}

document.getElementById("run").addEventListener("click", async () => {
  const key = keyInput.value.trim();
  if (!key) {
    show('<span class="bad">Paste your API key first.</span>');
    return;
  }
  chrome.storage.local.set({ key });
  show("Calling…");

  // A question shaped like the real ones: bounded, typed, over named state.
  const body = {
    state: {
      transcript: [
        { speaker: "interviewer", text: "How would you keep the counts accurate if a worker dies mid-batch?" },
        { speaker: "candidate", text: "So the dashboard reads from Redis, and it's really fast, users get a good experience." },
      ],
      latest_interviewer_question:
        "How would you keep the counts accurate if a worker dies mid-batch?",
    },
    questions: {
      answered_question: {
        type: "noul",
        instructions:
          "The interviewer asked `latest_interviewer_question`. Has the candidate addressed " +
          "that specific question? Answer yes if they addressed it, even partially. Answer " +
          "no if they spoke about a related but different topic.",
        criteria: {
          true: "The candidate engaged with the question that was actually asked.",
          false: "The candidate spoke about something adjacent but never addressed it.",
        },
      },
      current_phase: {
        type: "choice",
        instructions: "Which phase of a system design interview is `transcript` in?",
        criteria: {
          requirements: "Clarifying what the system must do.",
          high_level_design: "Naming components and how they connect.",
          scaling: "Handling growth, bottlenecks, failure.",
        },
      },
    },
  };

  const started = performance.now();
  try {
    const res = await fetch(ENDPOINT, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${key}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
    const ms = Math.round(performance.now() - started);
    const text = await res.text();

    if (!res.ok) {
      show(
        `<span class="bad">HTTP ${res.status}</span> in ${ms}ms\n\n${text}\n\n` +
          (text.includes("CORS")
            ? "Blocked by the origin allowlist. A backend proxy is required."
            : "Reached the API but the request was rejected, check the key.")
      );
      return;
    }

    const data = JSON.parse(text);
    const a = data.answers || {};
    show(
      `<span class="ok">It works. No backend needed.</span>\n\n` +
        `HTTP 200 in ${ms}ms · model ${data.model}\n` +
        `answered_question  noul ${a.answered_question?.noul}\n` +
        `current_phase      ${a.current_phase?.choice} ` +
        `(conf ${a.current_phase?.confidence})\n` +
        `tokens in/out      ${data.usage?.input_tokens}/${data.usage?.output_tokens}`
    );
  } catch (err) {
    // A CORS block surfaces here as an opaque TypeError, not as a status.
    show(
      `<span class="bad">fetch threw</span> after ${Math.round(performance.now() - started)}ms\n\n` +
        `${err}\n\nAn opaque "Failed to fetch" here means the browser blocked it ` +
        `before it left, CORS. Check host_permissions in manifest.json.`
    );
  }
});
