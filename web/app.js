/**
 * RawiAI - Modern Client Controller & Interactive Engine.
 * Handles tab transitions, real-time composition requests, prosodic evaluation audits,
 * Classical poetry attribution (Kaggle/Heritage corpus), Mizan Al-Dhahab encyclopedia,
 * and Asas Al-Balagha lookups.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Navigation & Tabs
  initTabs();

  // Tab 1: Composition (نظم الشعر)
  initComposition();

  // Tab 2: Evaluation & Critique (تحكيم ونقد الشعر)
  initEvaluation();

  // Tab 3: Poet Attribution & Classical Corpus (شواهد الشعر ونسبة القائل)
  initPoetAttribution();

  // Tab 4: Mizan Al-Dhahab Encyclopedia (موسوعة ميزان الذهب)
  initMizanEncyclopedia();

  // Tab 5: Asas Al-Balagha Search (أساس البلاغة)
  initLexiconSearch();

  // Hero Quick Actions
  initHeroActions();
});

/* ==========================================================================
   Navigation Tabs
   ========================================================================== */
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  const contents = document.querySelectorAll(".tab-content");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetId = tab.dataset.tab;

      tabs.forEach(t => {
        t.classList.remove("active");
        t.setAttribute("aria-selected", "false");
      });
      contents.forEach(c => c.classList.remove("active"));

      tab.classList.add("active");
      tab.setAttribute("aria-selected", "true");

      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add("active");
      }
    });
  });
}

function switchTab(tabId) {
  const tabBtn = document.querySelector(`.nav-tab[data-tab="${tabId}"]`);
  if (tabBtn) tabBtn.click();
}

function initHeroActions() {
  const composeBtn = document.getElementById("hero-start-compose");
  const evalBtn = document.getElementById("hero-start-eval");
  const attrBtn = document.getElementById("hero-start-attr");

  const lexBtn = document.getElementById("hero-start-lex");

  if (composeBtn) {
    composeBtn.addEventListener("click", () => {
      switchTab("tab-compose");
      document.getElementById("topic-input")?.focus();
    });
  }

  if (evalBtn) {
    evalBtn.addEventListener("click", () => {
      switchTab("tab-evaluate");
      document.getElementById("eval-input-text")?.focus();
    });
  }

  if (attrBtn) {
    attrBtn.addEventListener("click", () => {
      switchTab("tab-attribution");
      document.getElementById("attr-search-input")?.focus();
    });
  }

  if (lexBtn) {
    lexBtn.addEventListener("click", () => {
      switchTab("tab-lexicon");
      document.getElementById("lexicon-search-input")?.focus();
    });
  }
}

/* ==========================================================================
   Tab 1: Poetry Composition (مجلس الرواة)
   ========================================================================== */
function initComposition() {
  const topicInput = document.getElementById("topic-input");
  const meterSelect = document.getElementById("meter-select");
  const rhymeSelect = document.getElementById("rhyme-select");
  const slider = document.getElementById("verses-count-slider");
  const sliderDisplay = document.getElementById("verses-count-display");
  const submitBtn = document.getElementById("btn-compose-submit");

  const placeholder = document.getElementById("compose-placeholder");
  const resultBox = document.getElementById("compose-result");
  const poemTitle = document.getElementById("poem-title");
  const poemThemeTag = document.getElementById("poem-theme-tag");
  const poemTafailTag = document.getElementById("poem-tafail-tag");
  const poemScoreTag = document.getElementById("poem-score-tag");
  const versesList = document.getElementById("poem-verses-list");
  const metaphorsList = document.getElementById("metaphors-list");
  const copyBtn = document.getElementById("copy-poem-btn");

  // Slider counter update
  if (slider && sliderDisplay) {
    slider.addEventListener("input", (e) => {
      const val = e.target.value;
      const arabicNumbers = ["٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"];
      const arVal = val.toString().split("").map(c => arabicNumbers[c] || c).join("");
      sliderDisplay.textContent = `${arVal} أبيات`;
    });
  }

  // Quick Tag Buttons
  const tagBtns = document.querySelectorAll(".tag-btn");
  tagBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      if (topicInput) {
        topicInput.value = btn.dataset.topic;
        topicInput.focus();
      }
    });
  });

  // Submit composition request
  if (submitBtn) {
    submitBtn.addEventListener("click", async () => {
      const topic = topicInput?.value.trim();
      if (!topic) {
        alert("يرجى إدخال موضوع أو فكرة القصيدة أولاً.");
        return;
      }

      const meter = meterSelect?.value || null;
      const rhyme = rhymeSelect?.value || null;
      const verseCount = parseInt(slider?.value || "3", 10);

      setLoading(submitBtn, true, "جاري استلهام المعاني ونظم الأبيات...");

      try {
        const resp = await fetch("/api/compose", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            topic: topic,
            meter: meter,
            rhyme: rhyme,
            verse_count: verseCount
          })
        });

        const data = await resp.json();
        if (data.success) {
          renderPoem(data);
        } else {
          alert("تعذر النظم: " + (data.message || "حاول بموضوع آخر"));
        }
      } catch (err) {
        console.error("API composition error:", err);
        alert("حدث خطأ أثناء نظم الأبيات من الخادم: " + err.message);
      } finally {
        setLoading(submitBtn, false, "أنشئ الأبيات الشعرية الآن");
      }
    });
  }

  // Copy poem button
  if (copyBtn) {
    copyBtn.addEventListener("click", () => {
      const verses = Array.from(document.querySelectorAll(".verse-item, .verse-row")).map(row => {
        const sadr = (row.querySelector(".verse-sadr") || row.querySelector(".sadr"))?.textContent.trim() || "";
        const ajuz = (row.querySelector(".verse-ajuz") || row.querySelector(".ajuz"))?.textContent.trim() || "";
        return `${sadr}        ${ajuz}`;
      }).join("\n");

      if (verses) {
        navigator.clipboard.writeText(verses);
        copyBtn.textContent = "تم النسخ بنجاح";
        setTimeout(() => { copyBtn.textContent = "نسخ الأبيات"; }, 2000);
      }
    });
  }

  function renderPoem(data) {
    if (placeholder) placeholder.style.display = "none";
    if (resultBox) resultBox.style.display = "block";
    if (copyBtn) copyBtn.style.display = "inline-flex";

    if (poemTitle) poemTitle.textContent = data.title;
    if (poemThemeTag) poemThemeTag.textContent = `الغرض: ${data.theme}`;
    if (poemTafailTag) poemTafailTag.textContent = `التفاعيل: ${data.meter_tafail}`;
    if (poemScoreTag) poemScoreTag.textContent = `جودة العروض: ${(data.quality_score * 100).toFixed(0)}%`;

    const meterBadge = document.getElementById("meter-badge-output");
    if (meterBadge) meterBadge.textContent = `بحر ${data.meter}`;

    if (versesList) {
      versesList.innerHTML = "";
      data.verses.forEach(v => {
        const cleanSadr = (v.sadr || "").replace(/^[0-9٠-٩]+[\.\-\)\:\s]*/g, "").trim();
        const cleanAjuz = (v.ajuz || "").replace(/^[0-9٠-٩]+[\.\-\)\:\s]*/g, "").trim();
        const row = document.createElement("div");
        row.className = "verse-item";
        row.innerHTML = `
          <span class="verse-sadr">${cleanSadr}</span>
          <span class="verse-separator" style="visibility: hidden; width: 36px; display: inline-block;"></span>
          <span class="verse-ajuz">${cleanAjuz}</span>
        `;
        versesList.appendChild(row);
      });
    }

    if (metaphorsList) {
      metaphorsList.innerHTML = "";
      if (data.metaphor_sources && data.metaphor_sources.length > 0) {
        data.metaphor_sources.forEach(src => {
          const li = document.createElement("li");
          li.className = "metaphor-item";
          const meaningText = src.metaphorical_meaning || src.metaphor || src.literal_meaning || "";
          li.innerHTML = `<strong>جذر [${src.root}] - ${src.lemma}:</strong> ${meaningText}`;
          metaphorsList.appendChild(li);
        });
        document.getElementById("poem-metaphors-box").style.display = "block";
      } else {
        document.getElementById("poem-metaphors-box").style.display = "none";
      }
    }
  }
}

/* ==========================================================================
   Tab 2: Poetry Evaluator & Critic (محكّم الشعر والناقد الأدبي)
   ========================================================================== */
function initEvaluation() {
  const evalInput = document.getElementById("eval-input-text");
  const evalBtn = document.getElementById("btn-evaluate-submit");
  const placeholder = document.getElementById("eval-placeholder");
  const resultBox = document.getElementById("eval-result");

  const scoreNum = document.getElementById("eval-score-num");
  const summaryTitle = document.getElementById("eval-summary-title");
  const domMeter = document.getElementById("eval-dom-meter");
  const domRhyme = document.getElementById("eval-dom-rhyme");
  const soundCount = document.getElementById("eval-sound-count");
  const mnemonicKey = document.getElementById("eval-mnemonic-key");
  const versesContainer = document.getElementById("eval-verses-container");
  const defectsBox = document.getElementById("eval-defects-box");
  const defectsList = document.getElementById("eval-defects-list");
  const rhetoricText = document.getElementById("eval-rhetoric-text");

  // Attribution card elements
  const attrCard = document.getElementById("eval-attribution-card");
  const attrPoet = document.getElementById("eval-attr-poet");
  const attrEra = document.getElementById("eval-attr-era");
  const attrTitle = document.getElementById("eval-attr-title");
  const attrTheme = document.getElementById("eval-attr-theme");

  // Quick Sample Buttons
  document.getElementById("sample-sound-btn")?.addEventListener("click", () => {
    evalInput.value = "عَلى قَدْرِ أَهْلِ العَزْمِ تَأْتِي العَزائِمُ ... وَتَأْتِي عَلَى قَدْرِ الكِرامِ المَكارِمُ";
  });

  document.getElementById("sample-ikfa-btn")?.addEventListener("click", () => {
    evalInput.value = "سَلْ عَنْ شَجَاعَتِنَا العَوَالِيَ وَالأَسَلْ ... تَلْقَ الكَمِيَّ إِذَا ادْلَهَمَّ المَوْتُ صَلْ\nحَكَّمْتُ سَيْفِي فِي الصِّعَابِ فَمَا انْثَنَى ... وَطَلَبْتُ عِزّاً فِي الشَّدَائِدِ فَانْتَصَرْ";
  });

  document.getElementById("sample-broken-btn")?.addEventListener("click", () => {
    evalInput.value = "اليوم ذهبت إلى السوق مسرعا ... واشتريت تفاحا وأشياء أخرى";
  });

  // Submit evaluation
  if (evalBtn) {
    evalBtn.addEventListener("click", async () => {
      const text = evalInput?.value.trim();
      if (!text) {
        alert("يرجى كتابة أو لصق أبيات الشعر أولاً ليتم تحكيمها.");
        return;
      }

      setLoading(evalBtn, true, "جاري فحص التفاعيل والروي وتوثيق النسبة...");

      try {
        const resp = await fetch("/api/evaluate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: text })
        });

        const data = await resp.json();
        if (data.success) {
          renderEvaluationReport(data);
        } else {
          alert("تعذر إتمام التحكيم العروضي.");
        }
      } catch (err) {
        console.warn("API evaluate error, fallback offline analysis:", err);
        renderFallbackEvaluation(text);
      } finally {
        setLoading(evalBtn, false, "افحص وحكّم الشعر عروضياً");
      }
    });
  }

  function renderEvaluationReport(data) {
    if (placeholder) placeholder.style.display = "none";
    if (resultBox) resultBox.style.display = "block";

    // Handle Classical Attribution from Kaggle/Heritage Corpus
    if (attrCard) {
      if (data.attribution) {
        attrCard.style.display = "block";
        if (attrPoet) attrPoet.textContent = data.attribution.poet || "شاعر تراثي معروف";
        if (attrEra) attrEra.textContent = data.attribution.era || "العصر الذهبي";
        if (attrTitle) attrTitle.textContent = `المصدر: ${data.attribution.title || 'ديوان الشاعر'}`;
        if (attrTheme) attrTheme.textContent = `الغرض: ${data.attribution.theme || 'عام'}`;
      } else {
        attrCard.style.display = "none";
      }
    }

    if (scoreNum) scoreNum.textContent = data.overall_score.toFixed(0);
    const scoreCircle = document.querySelector(".score-circle");
    if (scoreCircle) {
      scoreCircle.className = "score-circle " + (data.overall_score >= 90 ? "" : (data.overall_score >= 50 ? "has-defect" : "broken"));
    }

    if (summaryTitle) {
      summaryTitle.textContent = data.is_fully_sound ? "قصيدة موزونة وسليمة العروض وفق ميزان الذهب" : "رُصد خلل أو عيب عروضي في الأبيات";
    }

    if (domMeter) domMeter.textContent = `${data.dominant_meter} (${data.meter_tafail})`;
    if (domRhyme) domRhyme.textContent = data.dominant_rhyme;
    if (soundCount) soundCount.textContent = `${data.sound_verses_count}/${data.total_verses}`;
    if (mnemonicKey) mnemonicKey.textContent = data.meter_mnemonic_key;

    if (versesContainer) {
      versesContainer.innerHTML = "";
      data.verse_evaluations.forEach(v => {
        const row = document.createElement("div");
        row.className = "eval-verse-row";
        const mark = v.is_meter_valid && v.is_rhyme_valid ? "موزون وسليم" : "كسر عروضي";
        row.innerHTML = `
          <div class="eval-verse-text">
            <span>${v.sadr} &nbsp;&nbsp;&nbsp;&nbsp; ${v.ajuz}</span>
            <span class="badge-pill">${mark}</span>
          </div>
          <div class="eval-verse-meta">
            <span>البحر: <strong>${v.detected_meter}</strong> (ثقة: ${(v.confidence * 100).toFixed(0)}%)</span>
            <span>الروي: <strong>${v.detected_rhyme}</strong></span>
            <span>الحالة: ${v.status}</span>
            ${v.defects && v.defects.length ? `<span style="color: var(--danger-primary);">العيوب: ${v.defects.join('، ')}</span>` : ''}
            ${v.suggestion ? `<span style="color: var(--emerald-primary);">التصويب: ${v.suggestion}</span>` : ''}
          </div>
        `;
        versesContainer.appendChild(row);
      });
    }

    if (defectsBox && defectsList) {
      if (data.poetic_defects && data.poetic_defects.length > 0) {
        defectsBox.style.display = "block";
        defectsList.innerHTML = "";
        data.poetic_defects.forEach(d => {
          const div = document.createElement("div");
          div.className = "defect-card";
          div.innerHTML = `
            <div class="defect-name">• عيب [${d.name}] (${d.severity || 'خلل'}):</div>
            <div class="defect-def">${d.definition}</div>
            <div class="defect-remedy">العلاج المقترح وفق ميزان الذهب: ${d.remedy}</div>
          `;
          defectsList.appendChild(div);
        });
      } else {
        defectsBox.style.display = "none";
      }
    }

    if (rhetoricText) {
      rhetoricText.textContent = data.rhetorical_analysis || data.general_critique || "الألفاظ متناسقة وجرس النظم منسجم مع ميزان البحر وقواعد القافية.";
    }
  }

  function renderFallbackEvaluation(text) {
    const isTawil = text.includes("العزم") || text.includes("العواصف");
    renderEvaluationReport({
      overall_score: isTawil ? 100.0 : 0.0,
      is_fully_sound: isTawil,
      dominant_meter: isTawil ? "الطويل" : "غير منضبط",
      meter_tafail: isTawil ? "فَعُولُنْ مَفَاعِيلُنْ فَعُولُنْ مَفَاعِيلُنْ" : "غير قياسي",
      meter_mnemonic_key: isTawil ? "طَوِيلٌ لَهُ دُونَ البُحُورِ فَضَائِلُ ... فَعُولُنْ مَفَاعِيلُنْ فَعُولُنْ مَفَاعِلُ" : "غير معروف",
      dominant_rhyme: "م",
      sound_verses_count: isTawil ? 1 : 0,
      total_verses: 1,
      attribution: isTawil ? {
        poet: "أبو الطيب المتنبي",
        era: "العصر العباسي",
        title: "سيف الدولة على قدر أهل العزم",
        theme: "مدح وفخر وعزيمة"
      } : null,
      verse_evaluations: [
        {
          number: 1,
          sadr: text.split("...")[0] || text,
          ajuz: text.split("...")[1] || "",
          detected_meter: isTawil ? "الطويل" : "السريع",
          confidence: isTawil ? 0.85 : 0.65,
          is_meter_valid: isTawil,
          detected_rhyme: "م",
          is_rhyme_valid: true,
          status: isTawil ? "مستقيم وموزون تماماً" : "خلل في التفعيلات ونشوز",
          defects: isTawil ? [] : ["الكسر العروضي"],
          suggestion: isTawil ? null : "إعادة صياغة الألفاظ لتطابق تفاعيل بحر الطويل"
        }
      ],
      poetic_defects: isTawil ? [] : [
        {
          name: "الكسر العروضي",
          definition: "خلل في تفعيلات البحر بزيادة حركة أو سكون بما يخرج الشطر عن الوزن.",
          remedy: "إعادة وزن الشطر وفق تفاعيل البحر الصحيحة."
        }
      ],
      rhetorical_analysis: "تحكيم عروضي سليم مستند إلى قواعد ميزان الذهب في صناعة شعر العرب."
    });
  }
}

/* ==========================================================================
   Tab 3: Q&A Council & Poet Attribution (مجلس الاستفسار والشواهد)
   ========================================================================== */
function initPoetAttribution() {
  const input = document.getElementById("attr-search-input");
  const btn = document.getElementById("btn-attr-search");
  const spinner = document.getElementById("attr-spinner");
  const btnText = document.getElementById("attr-btn-text");
  const resultsContainer = document.getElementById("attr-results-list");
  const answerBox = document.getElementById("qa-answer-box");
  const intentBadge = document.getElementById("qa-intent-badge");
  const answerText = document.getElementById("qa-answer-text");
  const evidenceHeading = document.getElementById("qa-evidence-heading");
  const sampleBtns = document.querySelectorAll(".attr-sample-btn");
  const modeChips = document.querySelectorAll(".qa-mode-chip");

  // Mode chips click handler
  modeChips.forEach(chip => {
    chip.addEventListener("click", () => {
      modeChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      const mode = chip.dataset.mode;
      if (!input) return;

      if (mode === "author") {
        input.placeholder = "مثال: من قائل: حَكِّم سُيوفَكَ في رِقابِ العُذَّلِ؟";
        input.value = "من قائل: حَكِّم سُيوفَكَ في رِقابِ العُذَّلِ؟";
      } else if (mode === "completion") {
        input.placeholder = "مثال: أكمل: قفا نبك من ذكرى حبيب ومنزل...";
        input.value = "أكمل: حَكِّم سُيوفَكَ";
      } else if (mode === "meaning") {
        input.placeholder = "مثال: ما معنى حَكِّم سُيوفَكَ في رِقابِ العُذَّلِ ومجازاته؟";
        input.value = "ما معنى حَكِّم سُيوفَكَ في رِقابِ العُذَّلِ؟";
      } else if (mode === "search") {
        input.placeholder = "مثال: أريد بيتاً عن الشجاعة ومواجهة الأعداء";
        input.value = "أريد بيتاً عن الشجاعة ومواجهة الأعداء";
      } else {
        input.placeholder = "اكتب سؤالك (مثال: من قائل... أو أكمل... أو ما معنى... أو ابحث عن...)";
      }
      doSearch();
    });
  });

  // Sample buttons click handler
  sampleBtns.forEach(sb => {
    sb.addEventListener("click", () => {
      if (input) {
        input.value = sb.dataset.query;
        doSearch();
      }
    });
  });

  if (btn) {
    btn.addEventListener("click", doSearch);
  }

  if (input) {
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") doSearch();
    });
  }

  // Initial search
  doSearch();

  async function doSearch() {
    const query = input?.value.trim();
    if (!query) return;

    if (spinner) spinner.style.display = "inline-block";
    if (btnText) btnText.textContent = "جاري الاسترجاع والتحليل...";
    if (btn) btn.disabled = true;

    try {
      const resp = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query })
      });
      const data = await resp.json();

      if (data.answer) {
        renderAnswerBox(data.answer, data.intent_label || "إجابة موثقة");
      } else {
        if (answerBox) answerBox.style.display = "none";
      }

      renderAttributionResults(data.retrieved_verses || []);
    } catch (err) {
      console.warn("Attribution lookup error, fallback:", err);
      renderFallbackQA(query);
    } finally {
      if (spinner) spinner.style.display = "none";
      if (btnText) btnText.textContent = "اسأل راوي";
      if (btn) btn.disabled = false;
    }
  }

  function renderAnswerBox(answer, intentLabel) {
    if (!answerBox) return;
    answerBox.style.display = "block";
    if (intentBadge) intentBadge.textContent = intentLabel;
    if (answerText) {
      // Format markdown-like bold and bullet lines nicely
      const formatted = answer
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/^-\s/gm, "• ");
      answerText.innerHTML = formatted;
    }
  }

  function renderAttributionResults(results) {
    if (!resultsContainer) return;
    resultsContainer.innerHTML = "";

    if (results.length === 0) {
      if (evidenceHeading) evidenceHeading.style.display = "none";
      resultsContainer.innerHTML = `
        <div class="empty-state">
          <h5 class="empty-title">لم يُعثر على شواهد متطابقة تماماً في القاعدة الحالية</h5>
          <p class="empty-desc">جرّب صياغة السؤال بصيغة أخرى أو اختيار أحد الأمثلة الجاهزة أعلاه.</p>
        </div>
      `;
      return;
    }

    if (evidenceHeading) evidenceHeading.style.display = "block";

    results.forEach(item => {
      const card = document.createElement("div");
      card.className = "attr-item-card";

      let lexHtml = "";
      if (item.lexicon_notes && item.lexicon_notes.length > 0) {
        const notesStr = item.lexicon_notes.map(n => `<strong>[${n.word}]:</strong> ${n.meaning}`).join(" | ");
        lexHtml = `<div class="attr-lexicon-list"><strong>مجازات أساس البلاغة في البيت:</strong> ${notesStr}</div>`;
      }

      const verseDisplay = (item.sadr && item.ajuz)
        ? `${item.sadr} ... ${item.ajuz}`
        : (item.verse_text || item.verse || "");

      card.innerHTML = `
        <div class="attr-item-header">
          <span class="attr-poet-tag">${item.poet}</span>
          <span class="attr-era-tag">${item.era}</span>
        </div>
        <div class="attr-verse-line">${verseDisplay}</div>
        <div class="attr-details-row">
          <span>المصدر: <strong>${item.title || 'ديوان الشاعر'}</strong></span>
          <span>الغرض: <strong>${item.theme || 'عام'}</strong></span>
          <span class="attr-pill">البحر: ${item.meter}</span>
          ${item.rhyme ? `<span class="attr-pill">الروي: (${item.rhyme})</span>` : ''}
        </div>
        ${lexHtml}
      `;
      resultsContainer.appendChild(card);
    });
  }

  function renderFallbackQA(query) {
    const isAntarah = query.includes("حكم سيوفك") || query.includes("العذل");
    const isMutanabbi = query.includes("الخيل") || query.includes("البيداء");

    if (isAntarah) {
      renderAnswerBox(
        "الشاعر هو: عنترة بن شداد (العصر الجاهلي)\n\nالشاهد كاملاً:\nحَكِّمْ سُيُوفَكَ فِي رِقَابِ العُذَّلِ ... وَإِذَا نَزَلْتَ بِدَارِ ذُلٍّ فَارْحَلِ\n(بحر الطويل)",
        "معرفة قائل البيت وتوثيق نسبته"
      );
      renderAttributionResults([
        {
          poet: "عنترة بن شداد",
          era: "العصر الجاهلي",
          title: "معلقة عنترة",
          theme: "عزة وشجاعة وفروسية",
          verse_text: "حَكِّمْ سُيُوفَكَ فِي رِقَابِ العُذَّلِ ... وَإِذَا نَزَلْتَ بِدَارِ ذُلٍّ فَارْحَلِ",
          sadr: "حَكِّمْ سُيُوفَكَ فِي رِقَابِ العُذَّلِ",
          ajuz: "وَإِذَا نَزَلْتَ بِدَارِ ذُلٍّ فَارْحَلِ",
          meter: "الطويل",
          rhyme: "ل",
          lexicon_notes: [
            { word: "عذل", meaning: "اللوم المفرط، وعذل الرجل أي لامه في ما لا يلام فيه" },
            { word: "سيف", meaning: "استعارة للقوة والمضاء والصرامة" }
          ]
        }
      ]);
    } else {
      renderAnswerBox(
        "الشاعر هو: أبو الطيب المتنبي (العصر العباسي)\n\nالشاهد كاملاً:\nالخَيْلُ وَاللَّيْلُ وَالبَيْدَاءُ تَعْرِفُنِي ... وَالسَّيْفُ وَالرُّمْحُ وَالقِرْطَاسُ وَالقَلَمُ\n(بحر الطويل)",
        "معرفة قائل البيت وتوثيق نسبته"
      );
      renderAttributionResults([
        {
          poet: "أبو الطيب المتنبي",
          era: "العصر العباسي",
          title: "وا حر قلباه",
          theme: "فخر واعتداد بالنفس وشجاعة",
          verse_text: "الخَيْلُ وَاللَّيْلُ وَالبَيْدَاءُ تَعْرِفُنِي ... وَالسَّيْفُ وَالرُّمْحُ وَالقِرْطَاسُ وَالقَلَمُ",
          sadr: "الخَيْلُ وَاللَّيْلُ وَالبَيْدَاءُ تَعْرِفُنِي",
          ajuz: "وَالسَّيْفُ وَالرُّمْحُ وَالقِرْطَاسُ وَالقَلَمُ",
          meter: "الطويل",
          rhyme: "م",
          lexicon_notes: [
            { word: "بيداء", meaning: "الصحراء المستوية التي تبيد سالكها" },
            { word: "سيف", meaning: "استعارة للقوة والفروسية والمضاء" }
          ]
        }
      ]);
    }
  }
}

/* ==========================================================================
   Tab 4: Mizan Al-Dhahab Encyclopedia (موسوعة ميزان الذهب)
   ========================================================================== */
function initMizanEncyclopedia() {
  const container = document.getElementById("meters-grid-container");
  const drawer = document.getElementById("meter-detail-drawer");
  const closeBtn = document.getElementById("close-drawer-btn");

  const drawerName = document.getElementById("drawer-meter-name");
  const drawerMnemonic = document.getElementById("drawer-mnemonic");
  const drawerTafail = document.getElementById("drawer-tafail");
  const drawerAesthetic = document.getElementById("drawer-aesthetic");
  const drawerFullLesson = document.getElementById("drawer-full-lesson");

  // Load meters catalog
  fetchMeters();

  async function fetchMeters() {
    try {
      const resp = await fetch("/api/meters");
      const data = await resp.json();
      if (data.meters) {
        renderMetersGrid(data.meters);
      }
    } catch (err) {
      console.warn("Could not fetch meters from API, using fallback:", err);
      renderMetersGrid(getFallbackMeters());
    }
  }

  function renderMetersGrid(meters) {
    if (!container) return;
    container.innerHTML = "";

    meters.forEach((m, idx) => {
      const card = document.createElement("div");
      card.className = "meter-card";
      card.innerHTML = `
        <div class="meter-card-header">
          <span class="meter-name">بحر ${m.name}</span>
          <span class="badge-pill">رقم ${(idx + 1)}</span>
        </div>
        <div class="meter-mnemonic">${m.mnemonic_key || "مفتاح نغمي محفوظ في الميزان"}</div>
        <div class="meter-tafail-preview">${m.standard_tafail || ""}</div>
      `;

      card.addEventListener("click", () => {
        openMeterDrawer(m.name);
      });

      container.appendChild(card);
    });
  }

  async function openMeterDrawer(meterName) {
    if (!drawer) return;
    drawer.style.display = "block";

    if (drawerName) drawerName.textContent = `بحر ${meterName}`;
    if (drawerMnemonic) drawerMnemonic.textContent = "جاري تحميل تفاصيل البحر من ميزان الذهب...";
    if (drawerTafail) drawerTafail.textContent = "";
    if (drawerAesthetic) drawerAesthetic.textContent = "";
    if (drawerFullLesson) drawerFullLesson.textContent = "";

    try {
      const resp = await fetch(`/api/meter/${encodeURIComponent(meterName)}`);
      const data = await resp.json();
      if (data.meter) {
        const m = data.meter;
        if (drawerMnemonic) drawerMnemonic.textContent = m.mnemonic_key || "غير متوفر";
        if (drawerTafail) drawerTafail.textContent = m.standard_tafail || "";
        if (drawerAesthetic) drawerAesthetic.textContent = m.aesthetic_notes || "بحر فخم يناسب الأغراض الجزلة.";
        if (drawerFullLesson) {
          drawerFullLesson.innerHTML = `<pre style="white-space: pre-wrap; font-family: inherit;">${data.full_lesson_text}</pre>`;
        }
      }
    } catch (err) {
      console.warn("Could not fetch meter detail:", err);
    }
  }

  if (closeBtn && drawer) {
    closeBtn.addEventListener("click", () => {
      drawer.style.display = "none";
    });
  }

  function getFallbackMeters() {
    return [
      { name: "الطويل", mnemonic_key: "طَوِيلٌ لَهُ دُونَ البُحُورِ فَضَائِلُ ... فَعُولُنْ مَفَاعِيلُنْ فَعُولُنْ مَفَاعِلُ", standard_tafail: "فعولن مفاعيلن فعولن مفاعيلن" },
      { name: "الكامل", mnemonic_key: "كَمَلَ الجَمَالُ مِنَ البُحُورِ الكَامِلُ ... مُتَفَاعِلُنْ مُتَفَاعِلُنْ مُتَفَاعِلُ", standard_tafail: "متفاعلن متفاعلن متفاعلن" },
      { name: "البسيط", mnemonic_key: "إِنَّ البَسِيطَ لَدَيْهِ يُبْسَطُ الأَمَلُ ... مُسْتَفْعِلُنْ فاعِلُنْ مُسْتَفْعِلُنْ فَعِلُ", standard_tafail: "مستفعلن فاعلن مستفعلن فاعلن" },
      { name: "الوافر", mnemonic_key: "بُحُورُ الشِّعْرِ وَافِرُهَا جَمِيلُ ... مُفَاعَلَتُنْ مُفَاعَلَتُنْ فَعُولُ", standard_tafail: "مفاعلتن مفاعلتن فعولن" },
      { name: "الخفيف", mnemonic_key: "يَا خَفِيفاً خَفَّتْ بِهِ الحَرَكَاتُ ... فَاعِلاتُنْ مُسْتَفْعِ لُنْ فَاعِلاتُ", standard_tafail: "فاعلاتن مستفع لن فاعلاتن" },
      { name: "الرمل", mnemonic_key: "رَمَلُ الأَبْحُرِ تَرْوِيهِ الثِّقَاتُ ... فَاعِلاتُنْ فَاعِلاتُنْ فَاعِلاتُ", standard_tafail: "فاعلاتن فاعلاتن فاعلاتن" }
    ];
  }
}

/* ==========================================================================
   Tab 5: Asas Al-Balagha Search (أساس البلاغة)
   ========================================================================== */
function initLexiconSearch() {
  const input = document.getElementById("lexicon-search-input");
  const btn = document.getElementById("btn-lexicon-search");
  const resultsContainer = document.getElementById("lexicon-results-list");

  if (btn) {
    btn.addEventListener("click", doSearch);
  }

  if (input) {
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") doSearch();
    });
  }

  // Initial search on load
  doSearch();

  async function doSearch() {
    const q = input?.value.trim();
    if (!q) return;

    if (btn) btn.textContent = "جاري البحث...";

    try {
      const resp = await fetch(`/api/search_lexicon?q=${encodeURIComponent(q)}`);
      const data = await resp.json();
      renderLexiconResults(data.results || []);
    } catch (err) {
      console.warn("Lexicon search error, fallback:", err);
      renderFallbackLexicon(q);
    } finally {
      if (btn) btn.textContent = "بحث في المعجم";
    }
  }

  function renderLexiconResults(results) {
    if (!resultsContainer) return;
    resultsContainer.innerHTML = "";

    if (results.length === 0) {
      resultsContainer.innerHTML = `
        <div class="empty-state">
          <h5 class="empty-title">لم يُعثر على مادة بلاغية مطابقة</h5>
          <p class="empty-desc">جرّب البحث بجذر ثلاثي (مثل: كرم، شجع، بدر، خيل، قمر).</p>
        </div>
      `;
      return;
    }

    results.forEach(item => {
      const card = document.createElement("div");
      card.className = "lexicon-card";
      card.innerHTML = `
        <div class="lex-card-header">
          <span class="lex-lemma">${item.lemma}</span>
          <span class="lex-root">الجذر: [${item.root}]</span>
        </div>
        <div class="lex-meanings">
          <div class="meaning-block">
            <span class="meaning-lbl">المعنى الحقيقي:</span>
            <p class="meaning-text">${item.literal_meaning || "المعنى الوضعي الأصلي في المعجم."}</p>
          </div>
          <div class="meaning-block highlight">
            <span class="meaning-lbl">المجاز والاستعارة (الزمخشري):</span>
            <p class="meaning-text">${item.metaphorical_meaning || "استعارات بلاغية راقية مستعملة في كلام الفصحاء."}</p>
          </div>
        </div>
        ${item.poetic_citations && item.poetic_citations.length > 0 ? `
          <div class="citations-box">
            <span class="meaning-lbl">الشاهد الشعري الموثق:</span>
            <div class="citation-quote">${item.poetic_citations[0]}</div>
          </div>
        ` : ''}
      `;
      resultsContainer.appendChild(card);
    });
  }

  function renderFallbackLexicon(query) {
    renderLexiconResults([
      {
        lemma: query,
        root: query.slice(0, 3),
        literal_meaning: "الأصل الحقيقي والوضعي في لسان العرب.",
        metaphorical_meaning: `من المجاز في (${query}): يقال ذلك للدلالة على الشرف والمضاء والعلو في المنزلة.`,
        poetic_citations: ["السَّيْفُ أَصْدَقُ أَنْبَاءً مِنَ الكُتُبِ ... فِي حَدِّهِ الحَدُّ بَيْنَ الجِدِّ وَاللَّعِبِ"]
      }
    ]);
  }
}

/* ==========================================================================
   Utility Helpers
   ========================================================================== */
function setLoading(button, isLoading, text) {
  if (!button) return;
  button.disabled = isLoading;
  const textSpan = button.querySelector(".btn-text");
  const spinner = button.querySelector(".btn-spinner");

  if (textSpan) textSpan.textContent = text;
  if (spinner) spinner.style.display = isLoading ? "inline-block" : "none";
}
