const $ = (id) => document.getElementById(id);
const status = $("status");

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
  try {
    await api("/stop", { method: "POST" });
    setStatus("Stopped.");
  } catch (e) {
    setStatus("Stop failed: " + e.message, true);
  }
};

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
