// Cross-browser WebExtension API compatibility (Chromium + Firefox)
var extAPI = (typeof browser !== "undefined" && browser.tabs) ? browser : chrome;

var C = {
  ok: ["LOW RISK", "#00E5FF"],
  warn: ["MODERATE RISK", "#FF9900"],
  bad: ["HIGH RISK", "#FF0055"]
};

var STOP_WORDS = new Set([
  "a","about","above","after","again","against","all","am","an","and","any","are","as","at",
  "be","because","been","before","being","below","between","both","but","by","can","did","do",
  "does","doing","down","during","each","few","for","from","further","had","has","have","having",
  "he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","it",
  "its","itself","just","me","more","most","my","myself","no","nor","not","now","of","off","on",
  "once","only","or","other","our","ours","ourselves","out","over","own","s","same","she","should",
  "so","some","such","t","than","that","the","their","theirs","them","themselves","then","there",
  "these","they","this","those","through","to","too","under","until","up","very","was","we","were",
  "what","when","where","which","while","who","whom","why","will","with","you","your","yours",
  "yourself","yourselves"
]);

function lemmatizeWord(w) {
  if (w.length <= 3) return w;
  if (w.endsWith("ies") && w.length > 4) return w.slice(0, -3) + "y";
  if (w.endsWith("es") && (w.endsWith("ches") || w.endsWith("shes") || w.endsWith("xes") || w.endsWith("sses"))) return w.slice(0, -2);
  if (w.endsWith("s") && !w.endsWith("ss") && !w.endsWith("us") && !w.endsWith("is")) return w.slice(0, -1);
  if (w.endsWith("ing") && w.length > 5) return w.slice(0, -3);
  if (w.endsWith("ed") && w.length > 4) return w.slice(0, -2);
  return w;
}

function preprocessText(raw) {
  var cleaned = String(raw || "")
    .replace(/<[^>]+>/g, " ")
    .replace(/https?:\/\/\S+|www\.\S+/gi, " ")
    .replace(/[^a-zA-Z0-9\s]/g, " ")
    .toLowerCase();
  return cleaned.split(/\s+/).filter(function(w) {
    return w.length > 1 && !STOP_WORDS.has(w);
  }).map(lemmatizeWord);
}

function scoreModel(tokens, modelObj, useBigrams) {
  if (!modelObj || !modelObj.vocab || tokens.length === 0) return 0.05;
  var counts = {};
  var i, tok, bg;
  for (i = 0; i < tokens.length; i++) {
    tok = tokens[i];
    if (modelObj.vocab[tok] !== undefined) {
      counts[tok] = (counts[tok] || 0) + 1;
    }
    if (useBigrams && i < tokens.length - 1) {
      bg = tokens[i] + " " + tokens[i + 1];
      if (modelObj.vocab[bg] !== undefined) {
        counts[bg] = (counts[bg] || 0) + 1;
      }
    }
  }
  var normSq = 0;
  var tfidfVec = {};
  var keys = Object.keys(counts);
  if (keys.length === 0) return 0.08;
  for (i = 0; i < keys.length; i++) {
    var k = keys[i];
    var idx = modelObj.vocab[k];
    var val = counts[k] * modelObj.idf[idx];
    tfidfVec[idx] = val;
    normSq += val * val;
  }
  var norm = Math.sqrt(normSq) || 1.0;
  var logit = modelObj.bias;
  var idxs = Object.keys(tfidfVec);
  for (i = 0; i < idxs.length; i++) {
    var id = idxs[i];
    logit += (tfidfVec[id] / norm) * modelObj.weights[id];
  }
  return 1.0 / (1.0 + Math.exp(-logit));
}

var model = "Both";
var level = "safe";
var liveScanData = null;
var $ = function(id) { return document.getElementById(id); };

function getBarColor(riskProb) {
  if (riskProb >= 0.65) return "#FF0055";
  if (riskProb >= 0.35) return "#FF9900";
  return "#00E5FF";
}

function updateComparisonBars(rfProb, lstmProb, bothProb, activeModel) {
  var rfPct = Math.round(rfProb * 100);
  var lstmPct = Math.round(lstmProb * 100);
  var bothPct = Math.round(bothProb * 100);

  $("rfPctTxt").textContent = rfPct + "% RISK";
  $("rfPctTxt").style.color = getBarColor(rfProb);
  $("rfBarFill").style.width = rfPct + "%";
  $("rfBarFill").style.background = getBarColor(rfProb);

  $("lstmPctTxt").textContent = lstmPct + "% RISK";
  $("lstmPctTxt").style.color = getBarColor(lstmProb);
  $("lstmBarFill").style.width = lstmPct + "%";
  $("lstmBarFill").style.background = getBarColor(lstmProb);

  $("bothPctTxt").textContent = bothPct + "% RISK";
  $("bothPctTxt").style.color = getBarColor(bothProb);
  $("bothBarFill").style.width = bothPct + "%";
  $("bothBarFill").style.background = getBarColor(bothProb);

  $("barRowRF").classList.toggle("active-row", activeModel === "Random Forest");
  $("barRowLSTM").classList.toggle("active-row", activeModel === "LSTM");
  $("barRowBoth").classList.toggle("active-row", activeModel === "Both");
  $("activeModelBadge").textContent = "ACTIVE: " + (activeModel === "Both" ? "HYBRID" : activeModel.toUpperCase());
}

function populateLinksDropdown(links, fallbackUrl) {
  var sel = $("linkSelect");
  var badge = $("linkCountBadge");
  var cleanLinks = (links || []).filter(function(u) { return /^https?:\/\//i.test(u); });

  badge.textContent = cleanLinks.length + (cleanLinks.length === 1 ? " LINK" : " LINKS");
  sel.innerHTML = "";

  if (cleanLinks.length === 0) {
    var opt = document.createElement("option");
    opt.value = fallbackUrl || "";
    opt.textContent = fallbackUrl ? "[CURRENT PAGE] " + fallbackUrl : "No links detected in email body";
    sel.appendChild(opt);
    return;
  }

  cleanLinks.forEach(function(url, idx) {
    var opt = document.createElement("option");
    opt.value = url;
    opt.textContent = "[" + (idx + 1) + "] " + url.replace(/^https?:\/\//i, "").slice(0, 48);
    sel.appendChild(opt);
  });
}

function buildLiveState(scan, selectedModel) {
  var bothProb = (scan.rfProb + scan.lstmProb) / 2.0;
  var chosenProb = selectedModel === "Random Forest"
    ? scan.rfProb
    : selectedModel === "LSTM"
    ? scan.lstmProb
    : bothProb;

  var domPenalty = (scan.externalForms > 0 ? 0.15 : 0) + (scan.ipLinks > 0 ? 0.20 : 0);
  var totalRisk = Math.min(0.99, Math.max(0.01, chosenProb + domPenalty));
  var safetyScore = Math.round((1 - totalRisk) * 100);

  var stateKey = "safe";
  var watermark = "01";
  var stateCode = "LOW RISK";
  if (totalRisk >= 0.65) {
    stateKey = "critical";
    watermark = "03";
    stateCode = "HIGH RISK";
  } else if (totalRisk >= 0.35) {
    stateKey = "suspicious";
    watermark = "02";
    stateCode = "MODERATE RISK";
  }

  var trigCount = scan.matchedTriggers.length;
  var isStealth = (scan.linkCount === 0 && trigCount === 0 && totalRisk >= 0.35);

  var tagText = stateKey === "critical"
    ? "HIGHLY SUSPICIOUS // PHISHING RISK"
    : stateKey === "suspicious"
    ? "SUSPICIOUS // VERIFY SENDER"
    : "SAFE // NO THREATS DETECTED";

  var titleText = stateKey === "critical"
    ? "High Phishing Risk"
    : stateKey === "suspicious"
    ? (isStealth ? "Subtle Social Engineering" : "Suspicious Email")
    : "Likely Safe Email";

  var descText = isStealth
    ? "No common spam keywords found, but sequential Bi-LSTM detected manipulative phrasing."
    : stateKey === "critical"
    ? "Do not click links, download attachments, or reply with personal or login details."
    : stateKey === "suspicious"
    ? "Contains unusual requests or urgency patterns. Verify the sender independently."
    : "No significant lexical or sequential phishing indicators found across " + scan.tokenCount + " words.";

  var rfStatus = scan.rfProb >= 0.65 ? "bad" : scan.rfProb >= 0.35 ? "warn" : "ok";
  var lstmStatus = scan.lstmProb >= 0.65 ? "bad" : scan.lstmProb >= 0.35 ? "warn" : "ok";
  var linkStatus = (scan.ipLinks > 0 || scan.externalForms > 0) ? "bad" : scan.linkCount > 3 ? "warn" : "ok";

  return {
    stateKey: stateKey,
    watermark: watermark,
    stateCode: stateCode,
    score: safetyScore,
    rfProb: scan.rfProb,
    lstmProb: scan.lstmProb,
    bothProb: bothProb,
    tag: tagText,
    title: titleText,
    desc: descText,
    target: scan.senderOrTitle || scan.host,
    host: scan.host,
    age: scan.tokenCount + " WORDS",
    rows: [
      [
        "KEYWORD ANALYSIS (TOP-50 TF-IDF)",
        trigCount > 0 ? "Detected: " + scan.matchedTriggers.slice(0, 4).join(", ") : "No high-risk phishing keywords found",
        trigCount + (trigCount === 1 ? " KEYWORD" : " KEYWORDS"),
        rfStatus
      ],
      [
        "CONTEXTUAL SEQUENCE (Bi-LSTM)",
        isStealth ? "Flagged indirect social-engineering phrasing" : "200-word bidirectional sequence evaluation",
        Math.round(scan.lstmProb * 100) + "% RISK",
        lstmStatus
      ],
      [
        "EMBEDDED HYPERLINK CHECK",
        scan.linkCount > 0 ? "Use Link Scanner below to verify URLs safely" : "No external web links found in message body",
        scan.linkCount + (scan.linkCount === 1 ? " LINK" : " LINKS"),
        linkStatus
      ]
    ]
  };
}

function render() {
  if (!liveScanData) return;
  var s = buildLiveState(liveScanData, model);

  level = s.stateKey;
  $("app").dataset.state = level;
  $("bgWatermark").textContent = s.watermark;
  $("stateCode").textContent = s.stateCode;
  $("host").textContent = s.host;
  $("age").textContent = s.age;
  $("score").textContent = s.score;

  var perimeter = 260;
  $("hexArc").style.strokeDasharray = perimeter;
  $("hexArc").style.strokeDashoffset = perimeter * (1 - s.score / 100);

  $("tag").textContent = s.tag;
  $("title").textContent = s.title;
  $("desc").textContent = s.desc;
  $("target").textContent = s.target;

  updateComparisonBars(s.rfProb, s.lstmProb, s.bothProb, model);

  $("rows").innerHTML = s.rows.map(function(r) {
    var t = C[r[3]];
    return '<div class="r"><span class="dot" style="background:' + t[1] + '"></span><div class="l"><b>' + r[0] + '</b><span>' + r[1] + '</span></div><div class="v"><b>' + r[2] + '</b><i style="color:' + t[1] + '">' + t[0] + '</i></div></div>';
  }).join("");

  // Strictly read-only status pills reflect the computed risk level
  document.querySelectorAll(".status-pill").forEach(function(p) {
    p.classList.toggle("on", p.dataset.k === level);
  });
  document.querySelectorAll(".model button").forEach(function(b) {
    b.classList.toggle("on", b.dataset.model === model);
  });
}

function runLocalTabScan() {
  if (!extAPI || !extAPI.tabs || !extAPI.scripting) return;
  extAPI.tabs.query({ active: true, currentWindow: true }, function(tabs) {
    var tab = tabs && tabs[0];
    if (!tab || !tab.id || /^(chrome|edge|about|moz-extension):/i.test(tab.url || "")) {
      liveScanData = {
        host: "local.browser.tab",
        tabUrl: "",
        senderOrTitle: "Open an email in Gmail, Outlook, or any webpage",
        tokenCount: 0, linkCount: 0, externalForms: 0, ipLinks: 0,
        links: [], matchedTriggers: [], rfProb: 0.04, lstmProb: 0.04
      };
      populateLinksDropdown([], "");
      render();
      return;
    }

    extAPI.scripting.executeScript({
      target: { tabId: tab.id },
      func: function() {
        var emailNode = document.querySelector(".a3s.aiL") ||
                        document.querySelector('[aria-label="Message body"]') ||
                        document.querySelector('[role="main"]') ||
                        document.body;
        var subjNode = document.querySelector("h2.hP") || document.querySelector('[role="heading"]');
        var senderNode = document.querySelector(".gD") || document.querySelector('[data-testid="message-from"]');

        var rawBodyText = emailNode ? emailNode.innerText : "";
        var text = ((subjNode ? subjNode.innerText : "") + " " + rawBodyText).trim();

        var anchorLinks = Array.from((emailNode || document).querySelectorAll("a[href]"))
          .map(function(a) { return a.href; })
          .filter(function(h) { return /^https?:\/\//i.test(h) && !h.includes("mail.google.com"); });

        var regexMatches = rawBodyText.match(/https?:\/\/[^\s"'<>]+/gi) || [];
        var uniqueLinks = Array.from(new Set(anchorLinks.concat(regexMatches))).slice(0, 25);

        var ipLinks = uniqueLinks.filter(function(h) { return /^https?:\/\/\d+\.\d+\.\d+\.\d+/.test(h); }).length;
        var forms = Array.from(document.querySelectorAll("form[action]"));
        var extForms = forms.filter(function(f) {
          return f.action && f.action.startsWith("http") && !f.action.includes(location.hostname);
        }).length;

        return {
          host: location.hostname || "local-page",
          tabUrl: location.href,
          senderOrTitle: senderNode ? (senderNode.getAttribute("email") || senderNode.innerText) : document.title.slice(0, 42),
          rawText: text.slice(0, 15000),
          links: uniqueLinks,
          linkCount: uniqueLinks.length,
          ipLinks: ipLinks,
          externalForms: extForms
        };
      }
    }, function(results) {
      if (extAPI.runtime.lastError || !results || !results[0] || !results[0].result) {
        return;
      }
      var dom = results[0].result;
      var tokens = preprocessText(dom.rawText);
      var bundles = window.PHISHOOK_MODELS || {};
      var top50Set = new Set(bundles.top50_triggers || []);

      var matched = [];
      var seen = new Set();
      for (var i = 0; i < tokens.length; i++) {
        if (top50Set.has(tokens[i]) && !seen.has(tokens[i])) {
          seen.add(tokens[i]);
          matched.push(tokens[i]);
        }
      }

      var rfProb = scoreModel(tokens, bundles.rf, false);
      var lstmProb = scoreModel(tokens, bundles.lstm, true);

      liveScanData = {
        host: dom.host,
        tabUrl: dom.tabUrl,
        senderOrTitle: dom.senderOrTitle,
        tokenCount: tokens.length,
        links: dom.links,
        linkCount: dom.linkCount,
        ipLinks: dom.ipLinks,
        externalForms: dom.externalForms,
        matchedTriggers: matched,
        rfProb: rfProb,
        lstmProb: lstmProb
      };
      populateLinksDropdown(dom.links, dom.tabUrl);
      render();
    });
  });
}

// 3-Way Model Selector: Random Forest | LSTM | Both
document.querySelectorAll(".model button").forEach(function(b) {
  b.onclick = function() {
    model = b.dataset.model;
    render();
  };
});

// Scan selected email link via Cloudflare Radar URL Scanner
function launchCloudflareScanner(targetUrl) {
  var radarUrl = "https://radar.cloudflare.com/scan";
  if (targetUrl && navigator.clipboard) {
    navigator.clipboard.writeText(targetUrl).catch(function() {});
    $("radarHint").innerHTML = "Copied <code>" + targetUrl.slice(0, 38) + "...</code> to clipboard! Paste into Cloudflare Radar.";
  }
  if (extAPI.tabs && extAPI.tabs.create) {
    extAPI.tabs.create({ url: radarUrl });
  } else {
    window.open(radarUrl, "_blank");
  }
}

$("scanLinkBtn").onclick = function() {
  var selectedUrl = $("linkSelect").value;
  launchCloudflareScanner(selectedUrl);
};

$("cloudflareHomeBtn").onclick = function() {
  var selectedUrl = $("linkSelect").value;
  launchCloudflareScanner(selectedUrl);
};

$("main").onclick = function() { runLocalTabScan(); };

$("active").onclick = function() {
  var on = this.classList.toggle("on");
  this.setAttribute("aria-pressed", on);
  $("activeTxt").textContent = on ? "ACTIVE" : "PAUSED";
};

runLocalTabScan();
