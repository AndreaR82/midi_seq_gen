const $ = (id) => document.getElementById(id);
const status = $("status");
let currentSeq = null;

function setStatus(msg, isError = false) {
  status.textContent = msg;
  status.className = isError ? "err" : "";
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

// Populate the MIDI channel dropdown (1-16).
for (let c = 1; c <= 16; c++) {
  const o = document.createElement("option");
  o.value = c;
  o.textContent = `Ch ${c}`;
  $("channel").appendChild(o);
}

async function loadPorts() {
  try {
    const { ports } = await api("/ports");
    const sel = $("port");
    sel.innerHTML = "";
    if (ports.length === 0) {
      const o = document.createElement("option");
      o.textContent = "(no MIDI ports found)";
      o.value = "";
      sel.appendChild(o);
    } else {
      for (const p of ports) {
        const o = document.createElement("option");
        o.value = p;
        o.textContent = p;
        sel.appendChild(o);
      }
    }
  } catch (e) {
    setStatus("Could not list ports: " + e.message, true);
  }
}

$("refresh").onclick = loadPorts;

$("generate").onclick = async () => {
  setStatus("Generating…");
  try {
    const seq = await api("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: $("query").value,
        bpm: Number($("bpm").value),
        key: $("key").value,
      }),
    });
    showSummary(seq);
    setArmed("generated sequence", seq);
    setStatus("Generated.");
  } catch (e) {
    setStatus("Generate failed: " + e.message, true);
  }
};

$("play").onclick = async () => {
  const port = $("port").value;
  if (!port) return setStatus("No MIDI port selected.", true);
  try {
    await api("/play", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ channel: Number($("channel").value), port_name: port }),
    });
    setStatus("Playing…");
  } catch (e) {
    setStatus("Play failed: " + e.message, true);
  }
};

$("stop").onclick = async () => {
  stopPreview();
  try {
    await api("/stop", { method: "POST" });
    setStatus("Stopped.");
  } catch (e) {
    setStatus("Stop failed: " + e.message, true);
  }
};

function setArmed(label, seq) {
  currentSeq = seq;
  $("armed").textContent = seq
    ? `Armed: ${label} — ${seq.loop_bars} bar(s), ${seq.notes.length} notes @ ${seq.bpm} BPM`
    : "";
}

function showSummary(seq) {
  const box = $("summary");
  box.hidden = false;
  box.innerHTML = "";
  const head = document.createElement("div");
  head.textContent = `${seq.notes.length} notes over ${seq.loop_bars} bar(s) @ ${seq.bpm} BPM`;
  const det = document.createElement("details");
  const sum = document.createElement("summary");
  sum.textContent = "raw JSON";
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(seq, null, 2);
  det.appendChild(sum);
  det.appendChild(pre);
  box.appendChild(head);
  box.appendChild(det);
}

loadPorts();


// --- Model melodies (.mid files written by melody_model.sample) -----------

async function loadMelodyList() {
  try {
    const { root, melodies } = await api("/melodies");
    $("melodyRoot").textContent = `Scanning ${root}`;
    const sel = $("melody");
    sel.innerHTML = "";
    if (melodies.length === 0) {
      const o = document.createElement("option");
      o.textContent = "(no .mid files found)";
      o.value = "";
      sel.appendChild(o);
      return;
    }
    for (const m of melodies) {
      const o = document.createElement("option");
      o.value = m.path;
      o.textContent = m.path;
      sel.appendChild(o);
    }
  } catch (e) {
    setStatus("Could not list melodies: " + e.message, true);
  }
}

$("refreshMelodies").onclick = loadMelodyList;

$("loadMelody").onclick = async () => {
  const path = $("melody").value;
  if (!path) return setStatus("No melody file selected.", true);
  try {
    const seq = await api("/melodies/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    showSummary(seq);
    setArmed(path, seq);
    setStatus(`Loaded ${path}. Play sends it out MIDI; Preview plays it here.`);
  } catch (e) {
    setStatus("Load failed: " + e.message, true);
  }
};

// --- Browser preview ------------------------------------------------------
// A plain oscillator synth, so you can audition without a Digitone or IAC
// port. Notes are scheduled against AudioContext.currentTime rather than with
// setTimeout, so the rhythm is sample-accurate instead of jittery.

let audioCtx = null;
let previewLoopId = null;
let previewVoices = [];

function midiToHz(pitch) {
  return 440 * Math.pow(2, (pitch - 69) / 12);
}

function scheduleNote(note, at, spb, loopBeats) {
  const start = at + note.start_beat * spb;
  // Truncate at the loop boundary, matching Player._run in app/player.py, so a
  // note that rings past the barline does not bleed into the next cycle.
  const endBeat = Math.min(note.start_beat + note.duration_beats, loopBeats);
  const dur = Math.max((endBeat - note.start_beat) * spb, 0.05);
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = "triangle";
  osc.frequency.value = midiToHz(note.pitch);
  const peak = 0.22 * (note.velocity / 127);
  // Short attack + release: enough shape to hear note boundaries, no clicks.
  gain.gain.setValueAtTime(0.0001, start);
  gain.gain.exponentialRampToValueAtTime(peak, start + 0.01);
  gain.gain.setValueAtTime(peak, start + dur * 0.8);
  gain.gain.exponentialRampToValueAtTime(0.0001, start + dur);
  osc.connect(gain).connect(audioCtx.destination);
  osc.start(start);
  osc.stop(start + dur + 0.02);
  previewVoices.push(osc);
  osc.onended = () => {
    previewVoices = previewVoices.filter((v) => v !== osc);
  };
}

function stopPreview() {
  if (previewLoopId !== null) {
    clearInterval(previewLoopId);
    previewLoopId = null;
  }
  for (const osc of previewVoices) {
    try { osc.stop(); } catch (_) {}
  }
  previewVoices = [];
}

async function startPreview(seq) {
  stopPreview();
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  await audioCtx.resume(); // browsers start the context suspended until a gesture

  const spb = 60 / seq.bpm;
  const loopBeats = seq.loop_bars * 4;
  const loopDur = loopBeats * spb;
  let nextLoopAt = audioCtx.currentTime + 0.1;

  // Lookahead scheduler: queue each loop shortly before it is due, so the
  // audio thread always has notes ahead of it and the loop seam stays tight.
  const pump = () => {
    while (nextLoopAt - audioCtx.currentTime < 0.3) {
      for (const n of seq.notes) {
        if (n.start_beat < loopBeats) scheduleNote(n, nextLoopAt, spb, loopBeats);
      }
      nextLoopAt += loopDur;
    }
  };
  pump();
  previewLoopId = setInterval(pump, 50);
}

$("preview").onclick = async () => {
  if (!currentSeq) return setStatus("Nothing armed — generate or load a melody first.", true);
  try {
    await startPreview(currentSeq);
    setStatus("Previewing in browser… (Stop to end)");
  } catch (e) {
    setStatus("Preview failed: " + e.message, true);
  }
};

loadMelodyList();
