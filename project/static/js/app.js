/**
 * Disease Prediction System — Main JavaScript
 * =============================================
 * Handles form submissions, API calls, image preview, result rendering,
 * Chart.js confidence charts, drag-&-drop upload, and dark-mode toggle.
 */

document.addEventListener("DOMContentLoaded", () => {

    // ─── Dark Mode Toggle ────────────────────────────────────────────
    const toggle = document.getElementById("darkModeToggle");
    const icon   = document.getElementById("darkModeIcon");
    const html   = document.documentElement;

    // Restore saved preference
    if (localStorage.getItem("theme") === "dark") {
        html.setAttribute("data-bs-theme", "dark");
        icon?.classList.replace("fa-moon", "fa-sun");
    }

    toggle?.addEventListener("click", () => {
        const dark = html.getAttribute("data-bs-theme") === "dark";
        html.setAttribute("data-bs-theme", dark ? "light" : "dark");
        icon?.classList.toggle("fa-moon", dark);
        icon?.classList.toggle("fa-sun", !dark);
        localStorage.setItem("theme", dark ? "light" : "dark");
    });

    // ─── Symptom Search Filters ──────────────────────────────────────
    const symptomSearch = document.getElementById("symptomSearch");
    symptomSearch?.addEventListener("input", (e) => {
        const val = e.target.value.toLowerCase().trim();
        const items = document.querySelectorAll("#symptomForm .symptom-check");
        items.forEach(item => {
            const labelText = item.querySelector("label").textContent.toLowerCase();
            if (labelText.includes(val)) {
                item.classList.remove("d-none");
            } else {
                item.classList.add("d-none");
            }
        });
    });

    const combSymptomSearch = document.getElementById("combSymptomSearch");
    combSymptomSearch?.addEventListener("input", (e) => {
        const val = e.target.value.toLowerCase().trim();
        const items = document.querySelectorAll("#combinedForm .symptom-check");
        items.forEach(item => {
            const labelText = item.querySelector("label").textContent.toLowerCase();
            if (labelText.includes(val)) {
                item.classList.remove("d-none");
            } else {
                item.classList.add("d-none");
            }
        });
    });

    // ─── Symptom Form ────────────────────────────────────────────────
    const symptomForm = document.getElementById("symptomForm");
    symptomForm?.addEventListener("submit", async (e) => {
        e.preventDefault();
        const checked = [...symptomForm.querySelectorAll("input[name='symptoms']:checked")]
                        .map(cb => cb.value);
        if (!checked.length) return alert("Please select at least one symptom.");

        show("symptomLoading"); hide("symptomResult");

        try {
            const res = await fetch("/predict_symptoms", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ symptoms: checked }),
            });
            const data = await res.json();
            hide("symptomLoading");
            if (data.error) return alert(data.error);

            setText("resPrediction", data.prediction);
            setText("resConfidence", data.confidence + " %");

            // Symptom badges
            const box = document.getElementById("resSymptoms");
            if (box) {
                box.innerHTML = (data.symptoms_used || checked)
                    .map(s => `<span class="badge bg-primary me-1 mb-1">${s.replace(/_/g, " ")}</span>`)
                    .join("");
            }

            // Description
            const descBox = document.getElementById("resDescBox");
            const descEl  = document.getElementById("resDescription");
            if (data.description && descBox && descEl) {
                descEl.textContent = data.description;
                descBox.classList.remove("d-none");
            }

            // Precautions
            const precBox = document.getElementById("resPrecBox");
            const precEl  = document.getElementById("resPrecautions");
            if (data.precautions && data.precautions.length && precBox && precEl) {
                precEl.innerHTML = data.precautions.map(p => `<li>${p}</li>`).join("");
                precBox.classList.remove("d-none");
            }

            // Demo note
            const note = document.getElementById("resNote");
            if (data.note && note) { note.textContent = data.note; note.classList.remove("d-none"); }

            // AI Cure Assistant
            if (data.cure_assistance) {
                renderCureAssistant("res", data.cure_assistance);
            }

            // Chart
            renderBarChart("confidenceChart", data.top_predictions || [
                { disease: data.prediction, confidence: data.confidence }
            ]);

            show("symptomResult");
        } catch (err) {
            hide("symptomLoading");
            alert("Request failed: " + err.message);
        }
    });

    // ─── Image Form ──────────────────────────────────────────────────
    const imageForm  = document.getElementById("imageForm");
    const imageInput = document.getElementById("imageInput");
    const imageBtn   = document.getElementById("imageBtn");

    imageInput?.addEventListener("change", () => {
        previewFile(imageInput, "imagePreview", "imagePreviewBox", "imageFileName");
        if (imageBtn) imageBtn.disabled = !imageInput.files.length;
    });

    // Drag & drop
    setupDragDrop("dropZone", imageInput, () => {
        previewFile(imageInput, "imagePreview", "imagePreviewBox", "imageFileName");
        if (imageBtn) imageBtn.disabled = false;
    });

    imageForm?.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!imageInput?.files.length) return alert("Please upload an image first.");

        show("imageLoading"); hide("imageResult");
        const fd = new FormData();
        fd.append("image", imageInput.files[0]);

        try {
            const res  = await fetch("/predict_image", { method: "POST", body: fd });
            const data = await res.json();
            hide("imageLoading");
            if (data.error) return alert(data.error);

            setText("imgDetected",   data.detected_symptom);
            setText("imgPrediction", data.prediction);
            setText("imgConfidence", data.confidence + " %");
            setImg("imgResultPreview", data.image_url);

            const note = document.getElementById("imgNote");
            if (data.note && note) { note.textContent = data.note; note.classList.remove("d-none"); }

            // AI Cure Assistant
            if (data.cure_assistance) {
                renderCureAssistant("img", data.cure_assistance);
            }

            show("imageResult");
        } catch (err) {
            hide("imageLoading");
            alert("Request failed: " + err.message);
        }
    });

    // ─── Combined Form ───────────────────────────────────────────────
    const combForm  = document.getElementById("combinedForm");
    const combInput = document.getElementById("combImageInput");

    combInput?.addEventListener("change", () => {
        previewFile(combInput, "combPreview", "combPreviewBox", "combFileName");
    });
    setupDragDrop("combDropZone", combInput, () => {
        previewFile(combInput, "combPreview", "combPreviewBox", "combFileName");
    });

    combForm?.addEventListener("submit", async (e) => {
        e.preventDefault();
        const checked = [...combForm.querySelectorAll("input[name='symptoms']:checked")]
                        .map(cb => cb.value);
        const hasImage = combInput?.files.length > 0;
        if (!checked.length && !hasImage) return alert("Select symptoms or upload an image.");

        show("combLoading"); hide("combResult");
        const fd = new FormData();
        checked.forEach(s => fd.append("symptoms", s));
        if (hasImage) fd.append("image", combInput.files[0]);

        try {
            const res  = await fetch("/predict_combined", { method: "POST", body: fd });
            const data = await res.json();
            hide("combLoading");
            if (data.error) return alert(data.error);

            const cp = data.combined_prediction || {};
            setText("combDisease",    cp.disease);
            setText("combConfidence", cp.confidence + " %");
            setText("combMethod",     cp.method);

            // Sub-results
            const sr = data.symptom_result;
            const ir = data.image_result;
            const symEl = document.getElementById("combSymRes");
            const imgEl = document.getElementById("combImgRes");

            if (symEl) symEl.innerHTML = sr
                ? `<strong>${sr.prediction || sr.disease}</strong> (${sr.confidence}%)`
                : "No symptoms selected.";
            if (imgEl) imgEl.innerHTML = ir
                ? `<strong>${ir.prediction || ir.disease}</strong> — ${ir.detected_symptom} (${ir.confidence}%)`
                : "No image uploaded.";

            // Chart
            const chartData = [];
            if (sr) chartData.push({ disease: "Symptoms: " + (sr.prediction||sr.disease), confidence: sr.confidence });
            if (ir) chartData.push({ disease: "Image: " + (ir.prediction||ir.disease), confidence: ir.confidence });
            chartData.push({ disease: "Combined: " + cp.disease, confidence: cp.confidence });
            renderBarChart("combChart", chartData);

            // AI Cure Assistant
            if (data.cure_assistance) {
                renderCureAssistant("comb", data.cure_assistance);
            }

            show("combResult");
        } catch (err) {
            hide("combLoading");
            alert("Request failed: " + err.message);
        }
    });

    // ─── Helper Functions ────────────────────────────────────────────

    function show(id) { document.getElementById(id)?.classList.remove("d-none"); }
    function hide(id) { document.getElementById(id)?.classList.add("d-none"); }
    function setText(id, txt) {
        const el = document.getElementById(id);
        if (el) el.textContent = txt ?? "—";
    }
    function setImg(id, src) {
        const el = document.getElementById(id);
        if (el && src) { el.src = src; el.classList.remove("d-none"); }
    }
    /** Render the AI Cure Assistant section with cure advice and product links */
    function renderCureAssistant(prefix, cureData) {
        const boxId = prefix + "CureAssistantBox";
        const box = document.getElementById(boxId);
        if (!box || !cureData) return;

        // Render cure advice
        const adviceEl = document.getElementById(prefix + "CureAdvice");
        if (adviceEl && cureData.cure_advice) {
            adviceEl.innerHTML = cureData.cure_advice
                .map(advice => `<li>${advice}</li>`)
                .join("");
        }

        // Render products
        const productsEl = document.getElementById(prefix + "Products");
        if (productsEl && cureData.products) {
            productsEl.innerHTML = cureData.products
                .map(product => `
                    <a href="${product.url}" target="_blank" rel="noopener noreferrer" class="product-card">
                        <span class="product-card-name">${product.name}</span>
                        <span class="product-card-type">${product.type}</span>
                        <span class="product-card-link">
                            <i class="fas fa-external-link-alt"></i> View on Amazon
                        </span>
                    </a>
                `)
                .join("");
        }

        // Render when to see doctor
        const doctorEl = document.getElementById(prefix + "WhenToSeeDoctor");
        if (doctorEl && cureData.when_to_see_doctor) {
            doctorEl.textContent = cureData.when_to_see_doctor;
        }

        // Render AI note/disclaimer
        const noteEl = document.getElementById(prefix + "AiNote");
        if (noteEl && cureData.ai_note) {
            const spanEl = noteEl.querySelector("span");
            if (spanEl) spanEl.textContent = cureData.ai_note;
        }

        // Show the cure assistant box
        box.classList.remove("d-none");
    }
    /** Preview a file input's first file in an <img> element. */
    function previewFile(input, imgId, boxId, nameId) {
        const file = input?.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
            const img = document.getElementById(imgId);
            if (img) img.src = ev.target.result;
            show(boxId);
        };
        reader.readAsDataURL(file);
        const nameEl = document.getElementById(nameId);
        if (nameEl) nameEl.textContent = file.name;
    }

    /** Wire drag-and-drop onto a zone element that forwards to a file input. */
    function setupDragDrop(zoneId, fileInput, onChange) {
        const zone = document.getElementById(zoneId);
        if (!zone || !fileInput) return;

        zone.addEventListener("click", () => fileInput.click());

        zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag-over"); });
        zone.addEventListener("dragleave", ()  => { zone.classList.remove("drag-over"); });
        zone.addEventListener("drop", (e) => {
            e.preventDefault();
            zone.classList.remove("drag-over");
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                onChange?.();
            }
        });
    }

    /** Render (or re-render) a horizontal bar chart with Chart.js. */
    let chartInstances = {};
    function renderBarChart(canvasId, items) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        // Destroy previous instance
        if (chartInstances[canvasId]) chartInstances[canvasId].destroy();

        const labels = items.map(i => i.disease);
        const values = items.map(i => i.confidence);
        
        const colors = [
            "rgba(79, 70, 229, 0.85)", // Indigo
            "rgba(13, 148, 136, 0.85)", // Teal
            "rgba(234, 179, 8, 0.85)",  // Gold
            "rgba(244, 63, 94, 0.85)",  // Rose
            "rgba(139, 92, 246, 0.85)"  // Violet
        ];
        const borderColors = [
            "#4f46e5",
            "#0d9488",
            "#eab308",
            "#f43f5e",
            "#8b5cf6"
        ];

        chartInstances[canvasId] = new Chart(ctx, {
            type: "bar",
            data: {
                labels,
                datasets: [{
                    label: "Confidence %",
                    data: values,
                    backgroundColor: items.map((_, i) => colors[i % colors.length]),
                    borderColor: items.map((_, i) => borderColors[i % borderColors.length]),
                    borderWidth: 1.5,
                    borderRadius: 8,
                }],
            },
            options: {
                indexAxis: "y",
                responsive: true,
                scales: { 
                    x: { 
                        beginAtZero: true, 
                        max: 100,
                        grid: { display: false },
                        ticks: { font: { family: "'Inter', sans-serif" } }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { font: { family: "'Inter', sans-serif", weight: 'bold' } }
                    }
                },
                plugins: { 
                    legend: { display: false },
                    tooltip: {
                        bodyFont: { family: "'Inter', sans-serif" },
                        titleFont: { family: "'Plus Jakarta Sans', sans-serif" }
                    }
                },
            },
        });
    }
});
