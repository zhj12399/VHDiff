"use strict";
(() => {
  const manifest = window.VHD_MANIFEST;
  const kind = document.body.dataset.gallery === "real" ? "r" : "g";
  const gallery = document.getElementById("gallery");
  const search = document.getElementById("search");
  const jump = document.getElementById("jump");
  const countFilter = document.getElementById("count-filter");
  const status = document.getElementById("results-status");
  const empty = document.getElementById("empty");
  let coverage = "all";
  const diseases = new Set();
  const eligible = manifest.conditions.filter(c => kind === "g" || c.status === "observed");
  const sections = new Map();
  const loading = new Map();
  const escape = value => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const full = code => manifest.conditions.find(c => c.code === code);

  function sampleMarkup(sample, condition) {
    const type = kind === "g" ? "Generated" : "Real";
    const title = `${type} ${condition.name} · example ${String(sample.example).padStart(2,"0")}`;
    return `<article class="sample"><div class="sample-top"><span class="sample-title">${type} example ${String(sample.example).padStart(2,"0")}</span><span class="sample-duration">3 s · 8 kHz WAV</span></div><audio controls preload="metadata" controlslist="nodownload" aria-label="Play ${escape(type.toLowerCase())} ${escape(condition.name)} example ${sample.example}" src="${sample.audio}"></audio><button class="plot-preview" type="button" data-plot-title="${escape(title)}" aria-label="Enlarge signal views for ${escape(title)}" title="Click to enlarge"><img src="${sample.plot}" alt="Waveform, Log-Mel spectrogram and 20 MFCC coefficients for ${escape(type.toLowerCase())} ${escape(condition.name)} example ${sample.example}" width="2220" height="1740" decoding="async"><span class="enlarge-icon" aria-hidden="true">⤢</span></button></article>`;
  }

  const viewer = document.createElement("dialog");
  viewer.className = "plot-dialog";
  viewer.setAttribute("aria-labelledby", "plot-dialog-title");
  viewer.innerHTML = '<header class="plot-dialog-head"><h2 id="plot-dialog-title"></h2><button class="plot-close" type="button" aria-label="Close enlarged signal views">×</button></header><div class="plot-dialog-body"><img alt=""></div>';
  document.body.appendChild(viewer);
  const viewerImage = viewer.querySelector("img");
  gallery.addEventListener("click", event => {
    const button = event.target.closest(".plot-preview");
    if (!button) return;
    const image = button.querySelector("img");
    viewer.querySelector("h2").textContent = button.dataset.plotTitle;
    viewerImage.src = image.src;
    viewerImage.alt = image.alt;
    viewer.showModal();
    document.body.classList.add("plot-open");
  });
  viewer.querySelector(".plot-close").addEventListener("click", () => viewer.close());
  viewer.addEventListener("click", event => { if (event.target === viewer) viewer.close(); });
  viewer.addEventListener("close", () => { document.body.classList.remove("plot-open"); viewerImage.removeAttribute("src"); });

  function loadCondition(code) {
    const section = sections.get(code);
    if (section.dataset.loaded === "true") return Promise.resolve();
    if (loading.has(code)) return loading.get(code);
    const condition = full(code);
    const key = `${kind}-${code}`;
    const promise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = (kind === "g" ? condition.generatedPack : condition.realPack) + "?v=20261012-ts2";
      script.onload = () => {
        const samples = window.VHD_SAMPLE_DATA && window.VHD_SAMPLE_DATA[key];
        if (!samples || samples.length !== 2) { reject(new Error("Incomplete sample pack")); return; }
        section.querySelector(".samples").innerHTML = samples.map(s => sampleMarkup(s, condition)).join("");
        section.dataset.loaded = "true";
        script.remove();
        resolve();
      };
      script.onerror = () => { script.remove(); reject(new Error("Audio samples could not be loaded")); };
      document.head.appendChild(script);
    }).catch(() => {
      section.querySelector(".samples").innerHTML = '<div class="load-error">Samples could not be loaded. <button type="button" class="retry">Try again</button></div>';
      loading.delete(code);
      section.querySelector(".retry").addEventListener("click", () => loadCondition(code));
    });
    loading.set(code, promise);
    return promise;
  }

  eligible.forEach(condition => {
    const isNew = condition.status === "unobserved";
    const section = document.createElement("section");
    section.className = "condition";
    section.id = `condition-${condition.code}`;
    section.dataset.code = condition.code;
    const badge = kind === "r" ? "Real database audio" : isNew ? "Absent from database" : "Observed in database";
    const support = isNew ? '<strong>0</strong> database recordings' : `<strong>${condition.databaseRecordings}</strong> database recording${condition.databaseRecordings === 1 ? "" : "s"}`;
    const subtitle = condition.fullNames.length ? condition.fullNames.join(" · ") : "All five disease attributes absent";
    section.innerHTML = `<header class="condition-head"><div><div class="condition-title"><h3>${escape(condition.name)}</h3><span class="badge ${isNew ? "new" : ""}">${badge}</span></div><p class="condition-sub"><span class="code">${condition.code}</span>${escape(subtitle)}</p></div><div class="support">${support}</div></header><div class="samples"><div class="loading">Loading two audio examples…</div><div class="loading" aria-hidden="true">Loading two audio examples…</div></div>`;
    gallery.appendChild(section);
    sections.set(condition.code, section);
    const option = document.createElement("option");
    option.value = condition.code;
    option.textContent = condition.name + (isNew ? " · absent" : "");
    jump.appendChild(option);
  });

  const observer = "IntersectionObserver" in window ? new IntersectionObserver(entries => {
    entries.forEach(entry => { if (entry.isIntersecting && !entry.target.hidden) loadCondition(entry.target.dataset.code); });
  }, {rootMargin: "450px 0px"}) : null;
  sections.forEach(section => observer ? observer.observe(section) : loadCondition(section.dataset.code));

  function filter() {
    const query = search.value.trim().toLowerCase().replace(/\s*\+\s*/g, " + ");
    const requestedCount = countFilter.value;
    let shown = 0;
    eligible.forEach(condition => {
      const haystack = `${condition.name} ${condition.code} ${condition.fullNames.join(" ")}`.toLowerCase();
      const matches = (!query || haystack.includes(query)) &&
        (coverage === "all" || condition.status === coverage) &&
        (requestedCount === "all" || condition.diseases.length === Number(requestedCount)) &&
        [...diseases].every(d => condition.diseases.includes(d));
      const section = sections.get(condition.code);
      section.hidden = !matches;
      if (matches) {
        shown++;
        if (section.getBoundingClientRect().top < window.innerHeight + 450) loadCondition(condition.code);
      } else section.querySelectorAll("audio").forEach(audio => audio.pause());
    });
    status.textContent = `${shown} configuration${shown === 1 ? "" : "s"} · ${shown * 2} ${kind === "g" ? "generated" : "real"} audio examples`;
    empty.hidden = shown !== 0;
  }
  search.addEventListener("input", filter);
  countFilter.addEventListener("change", filter);
  document.querySelectorAll("[data-coverage]").forEach(button => button.addEventListener("click", () => {
    coverage = button.dataset.coverage;
    document.querySelectorAll("[data-coverage]").forEach(b => b.setAttribute("aria-pressed", String(b === button)));
    filter();
  }));
  document.querySelectorAll("[data-disease]").forEach(button => button.addEventListener("click", () => {
    const disease = button.dataset.disease;
    if (diseases.has(disease)) diseases.delete(disease); else diseases.add(disease);
    button.setAttribute("aria-pressed", String(diseases.has(disease)));
    filter();
  }));
  function reset() {
    search.value = ""; countFilter.value = "all"; coverage = "all"; diseases.clear();
    document.querySelectorAll("[data-disease]").forEach(b => b.setAttribute("aria-pressed", "false"));
    document.querySelectorAll("[data-coverage]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.coverage === "all")));
    filter();
  }
  document.getElementById("reset").addEventListener("click", reset);
  jump.addEventListener("change", () => {
    if (!jump.value) return;
    reset();
    const section = sections.get(jump.value);
    loadCondition(jump.value);
    section.scrollIntoView({behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth"});
  });
  document.addEventListener("play", event => {
    if (event.target.tagName === "AUDIO") document.querySelectorAll("audio").forEach(audio => { if (audio !== event.target) audio.pause(); });
  }, true);
  document.getElementById("back-top").addEventListener("click", () => window.scrollTo({top:0,behavior:"smooth"}));
  filter();
  const hashCode = window.location.hash.replace("#condition-", "");
  if (sections.has(hashCode)) {
    jump.value = hashCode;
    loadCondition(hashCode).then(() => sections.get(hashCode).scrollIntoView());
  }
})();
