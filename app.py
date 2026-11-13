from __future__ import annotations

import csv
from html import escape
import io
import json
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from src.inference import create_ocr_engine, run_ocr
from src.detection import classify_template_side


FIELD_LABELS_AR = {
    "name": "الاسم",
    "first_name": "الاسم الأول",
    "paternal_names": "باقي الاسم",
    "address": "العنوان",
    "governorate": "المحافظة",
    "birth_date": "تاريخ الميلاد",
    "national_id": "الرقم القومي",
    "serial_number": "الرقم التسلسلي",
    "issue_date": "تاريخ الإصدار",
    "occupation": "المهنة",
    "gender": "النوع",
    "religion": "الديانة",
    "marital_status": "الحالة الاجتماعية",
    "expiry_date": "تاريخ انتهاء البطاقة",
}

st.set_page_config(page_title="Egyptian ID OCR", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root {
  --ink: #12233f;
  --muted: #5b6b84;
  --blue: #173b70;
  --line: #cbd7e8;
  --surface: rgba(255,255,255,.86);
  --soft: #eaf1fc;
}
[data-testid="stAppViewContainer"] {
  background: radial-gradient(ellipse at 10% 0%, #f8fbff 0, #eff4fc 55%, #e9f0fa 100%);
  color: var(--ink);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] {
  max-width: 1480px;
  padding: 1.5rem clamp(1rem, 3vw, 2.5rem) 2.5rem;
}
section[data-testid="stSidebar"], [data-testid="collapsedControl"] { display:none!important; }
.app-hero { display:flex; direction:rtl; text-align:right; gap:1rem; align-items:stretch; margin: .25rem 0 1.4rem; }
.app-hero-mark { width:8px; min-width:8px; border-radius:8px; background:#285ca4; }
.app-hero h1 { color:var(--ink); font-size:clamp(1.8rem,3.5vw,2.7rem); line-height:1.08; margin:0; font-weight:780; }
.app-hero p { color:#24334b; font-size:clamp(1rem,1.6vw,1.2rem); margin:.45rem 0 0; }
.upload-title { color:var(--ink); font-size:1.1rem; font-weight:750; margin:.25rem 0 .65rem; }
[data-testid="stFileUploaderDropzone"] { border:1.5px dashed #92a7c7; border-radius:16px; background:rgba(255,255,255,.82); min-height:112px; }
[data-testid="stFileUploaderDropzone"]:hover { border-color:#285ca4; background:#f6f9ff; }
[data-testid="stBaseButton-primary"] { background:linear-gradient(115deg,#24558e,#112b58); border:0; border-radius:12px; min-height:46px; font-weight:700; }
[data-testid="stBaseButton-secondary"] { border:1px solid #9fb2cf; border-radius:12px; color:#173b70; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--line)!important; border-radius:16px!important; background:var(--surface); box-shadow:0 4px 18px rgba(31,58,96,.05); }
[data-testid="stImage"] img { border:1px solid #ced8e7; border-radius:16px; }
.section-title { color:var(--ink); font-weight:760; font-size:1.18rem; border-bottom:1px solid #d5dfed; padding-bottom:.55rem; margin:.2rem 0 .8rem; }
.field-value { color:#13243f; font-size:1rem; font-weight:650; line-height:1.6; overflow-wrap:anywhere; }
.confidence-track { height:8px; width:100%; background:#dbe5f3; border-radius:99px; overflow:hidden; margin-top:.55rem; }
.confidence-fill { height:100%; background:linear-gradient(90deg,#2d63ad,#173b70); border-radius:99px; }
.confidence-label { color:var(--muted); font-size:.82rem; }
/* Keep Arabic copy consistently right-to-left and readable on Streamlit's light theme. */
[dir="rtl"] { direction:rtl; text-align:right; unicode-bidi:plaintext; }
[data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] { color:#172b49; }
[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"],
[data-testid="stFileUploaderDropzone"] button { color:#173b70!important; }
@media (max-width: 900px) {
  [data-testid="stMainBlockContainer"] { padding:1rem .9rem 2rem; }
  [data-testid="stHorizontalBlock"] { flex-wrap:wrap; gap:.7rem; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width:min(100%, 320px); flex:1 1 320px; }
  .app-hero { margin-bottom:1rem; }
}
@media (max-width: 520px) {
  .app-hero h1 { font-size:1.7rem; }
  .app-hero p { font-size:.96rem; }
  [data-testid="stFileUploaderDropzone"] { min-height:96px; padding:.75rem; }
  [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] { width:100%; }
  [data-testid="stImage"] img { border-radius:12px; }
}
</style>
<div class="app-hero">
  <div class="app-hero-mark"></div>
  <div><h1 dir="rtl">استخراج بيانات البطاقة الشخصية المصرية</h1></div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_ocr_engine(language: str, textline_orientation: bool):
    return create_ocr_engine(language=language, model_dir=None, device="cpu", detector_size="mobile", textline_orientation=textline_orientation)


def visualize_boxes(image: Image.Image, lines: list[dict[str, Any]]) -> Image.Image:
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    for index, line in enumerate(lines, 1):
        box = line.get("bbox")
        if not box:
            continue
        if len(box) == 4 and all(isinstance(value, (int, float)) for value in box):
            draw.rectangle(tuple(map(float, box)), outline=(220, 38, 38), width=3)
            x, y = float(box[0]), float(box[1])
        else:
            points = [(float(point[0]), float(point[1])) for point in box]
            draw.polygon(points, outline=(220, 38, 38), width=3)
            x, y = points[0]
        draw.text((x, max(0, y - 16)), str(index), fill=(220, 38, 38), stroke_width=2, stroke_fill="white")
    return marked


def result_csv(result: dict[str, Any]) -> str:
    output = io.StringIO(newline="")
    columns = ["kind", "field_name", "raw_text", "processed_value", "confidence", "bbox", "review_status", "validation_status"]
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    if result.get("fields"):
        for name, field in result["fields"].items():
            validation = field.get("validation_result") or {}
            writer.writerow({
                "kind": "field_candidate",
                "field_name": name,
                "raw_text": field.get("raw_ocr_output"),
                "processed_value": field.get("processed_value"),
                "confidence": field.get("confidence"),
                "bbox": json.dumps(field.get("bbox"), ensure_ascii=False),
                "review_status": field.get("review_status"),
                "validation_status": validation.get("status"),
            })
    else:
        for line in result.get("raw_lines", []):
            writer.writerow({"kind": "raw_line", "raw_text": line.get("raw_text"), "confidence": line.get("confidence"), "bbox": json.dumps(line.get("bbox"), ensure_ascii=False), "review_status": "needs_human_review"})
    return output.getvalue()


def render_region_candidate(field_name: str, candidate: dict[str, Any], image: Image.Image) -> None:
    label = FIELD_LABELS_AR.get(field_name, field_name)
    value = candidate.get("field_value_candidate")
    confidence = candidate.get("confidence")
    with st.container(border=True):
        title_col, value_col = st.columns([1, 2.6], gap="small")
        with title_col:
            st.markdown(f"**{label}**")
        with value_col:
            st.markdown(f'<div class="field-value">{escape(str(value or "غير مقروء — راجعي الصورة"))}</div>', unsafe_allow_html=True)
        status = candidate.get("field_candidate_status", "تحتاج مراجعة")
        if status.startswith("derived_from_national_id"):
            status = "مستنتج من الرقم القومي · يحتاج تأكيدًا بصريًا"
        st.caption(status)
        if candidate.get("candidate_provenance") == "full_image_text_boxes_position_grouping":
            st.caption("مُجمّع من سطور الكشف الكاملة بحسب موضعها؛ راجعي الصندوق على الصورة.")
        elif candidate.get("candidate_provenance") == "occupation_tight_crop":
            st.caption("مرشح من قصّة بديلة أسفل سطر الرقم؛ راجعي النص والقصّة قبل اعتماده.")
        if candidate.get("candidate_warning"):
            st.warning(candidate["candidate_warning"])
        if confidence is not None:
            score = max(0.0, min(1.0, float(confidence)))
            st.markdown(f'<div class="confidence-track"><div class="confidence-fill" style="width:{score * 100:.1f}%"></div></div>', unsafe_allow_html=True)
            st.caption(f"درجة النموذج: {confidence:.3f} · ليست مقياسًا لصحة الحقل")
        else:
            st.caption("لم ينتج النموذج درجة ثقة")
        with st.expander("النص الخام والقصّة"):
            st.write(f"النص الخام: {candidate.get('raw_ocr_output') or 'لم يُكتشف نص'}")
            ordered = candidate.get("reading_order_candidate")
            if ordered and ordered != candidate.get("raw_ocr_output"):
                st.write(f"مرشح ترتيب الأرقام: **{ordered}**")
            if candidate.get("formatted_display_candidate"):
                st.write(f"تنسيق عرض محتمل (لا يغيّر الخام): **{candidate['formatted_display_candidate']}**")
            id_date = candidate.get("national_id_date_candidate") or {}
            if id_date.get("value"):
                st.info(f"تاريخ مستنتج من الرقم القومي: **{id_date['value']}** · راجعي الرقم والصورة")
                if candidate.get("printed_date_candidate"):
                    st.caption(f"التاريخ المقروء من الصورة: {candidate['printed_date_candidate']}")
            elif candidate.get("date_format_status"):
                date_status = candidate["date_format_status"]
                if date_status.get("status", "").startswith("invalid") or date_status.get("status") == "incomplete_month":
                    st.warning(date_status["reason"])
                else:
                    st.caption(date_status["reason"])
            for retry in candidate.get("preprocessing_retry_candidates", []):
                st.caption(f"مرشح إعادة القراءة · {retry['preprocessing']}: {retry['raw_text']} · ثقة {retry['confidence']}")
                if retry.get("bbox"):
                    st.image(image.crop(tuple(retry["bbox"])), caption="قصّة المهنة البديلة", use_container_width=True)
            alternate = candidate.get("alternate_region_candidate")
            if alternate:
                st.caption(f"مرشح القصّة التقريبية البديل: {alternate.get('raw_ocr_output')} · ثقة {alternate.get('confidence')}")
                if alternate.get("bbox"):
                    st.image(image.crop(tuple(alternate["bbox"])), caption="القصّة التقريبية للمقارنة", use_container_width=True)
            bbox = candidate.get("bbox")
            if bbox and len(bbox) == 4:
                st.image(image.crop(tuple(bbox)), caption=f"قصّة الصورة · {label} · {bbox}", use_container_width=True)


mode = "Run OCR on a new image"

if mode == "Run OCR on a new image":
    with st.container(border=True):
        st.markdown('<div class="upload-title" dir="rtl">ارفع صورة البطاقة الشخصية</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("اسحب الصورة إلى هنا أو اختر ملفًا من جهازك", type=["png", "jpg", "jpeg", "tif", "tiff", "webp"], help="المعالجة تتم محليًا. للحصول على قصّات أدق، اجعل البطاقة كاملة وواضحة داخل الصورة.", label_visibility="collapsed")
    control_cols = st.columns([1, 1, 1.3], gap="medium")
    with control_cols[0]:
        side_choice = st.selectbox("وجه البطاقة", ["front", "back", "unknown", "auto"], format_func=lambda x: {"front":"الأمامي", "back":"الخلفي", "unknown":"غير محدد", "auto":"تلقائي (قالب تجريبي)"}[x], help="لصور البطاقات الحقيقية اختاري الوجه يدويًا. الاختيار التلقائي مخصص لقالب البيانات التجريبي فقط.")
    with control_cols[1]:
        language = st.selectbox("لغة التعرّف", ["ar", "la"], format_func=lambda x: "عربي وأرقام" if x == "ar" else "لاتيني وأرقام")
    with control_cols[2]:
        st.empty()
    if uploaded is not None:
        try:
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(uploaded.getvalue()))).convert("RGB")
            # Cloud instances have tight RAM limits. Keep enough pixels for OCR
            # while avoiding oversized full-image tensors and crop batches.
            image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        except (UnidentifiedImageError, OSError) as exc:
            st.error(f"Could not decode image: {exc}")
        else:
            with st.expander(f"معاينة الصورة قبل التشغيل · {image.width}×{image.height}", expanded=False):
                st.image(image, caption="الصورة الأصلية دون تعديل", use_container_width=True)
            template_mapping_confirmed = False
            if image.size == (1080, 1080):
                template_mapping_confirmed = st.checkbox("I confirm this image uses the exact EGID v1 template on a 1080×1080 canvas", value=False)
            if side_choice == "auto":
                side_result = classify_template_side(image)
                effective_side = side_result["side"]
                side_source = side_result["source"] if effective_side != "unknown" else "unknown"
                if effective_side == "unknown":
                    st.info("Side is unknown for this canvas or barcode score. Choose a side manually if you know it.")
                else:
                    st.info(f"Template side heuristic: {effective_side} · barcode score {side_result['score']:.3f}. This rule is specific to the supplied synthetic template.")
            else:
                effective_side = side_choice
                side_source = "user_selected" if side_choice != "unknown" else "unknown"
            try_front_regions = False
            if effective_side == "front" and not (template_mapping_confirmed and image.size == (1080, 1080)):
                try_front_regions = st.checkbox(
                    "Try Egyptian front-side field regions (approximate layout)",
                    value=True,
                    help="Useful when a photographed Egyptian ID card nearly fills the image. It runs OCR on enlarged name, address, date, national ID and serial regions. Inspect each crop and prediction; other card layouts may not fit this profile.",
                )
            try_back_regions = False
            if effective_side == "back" and not (template_mapping_confirmed and image.size == (1080, 1080)):
                try_back_regions = st.checkbox(
                    "Try Egyptian back-side field regions (approximate layout)",
                    value=True,
                    help="Runs OCR on enlarged regions for national ID, issue date, occupation, gender, religion, marital status and expiry date. The profile is approximate; compare each crop to the original image.",
                )
            if st.button("Extract", type="primary"):
                with tempfile.TemporaryDirectory(prefix="egid_ocr_") as temp_dir:
                    temp_path = Path(temp_dir) / "input.png"
                    image.save(temp_path, format="PNG", optimize=True)
                    try:
                        with st.spinner("Running OCR and collecting raw line predictions…"):
                            use_regions = try_front_regions or try_back_regions
                            model_dir = None
                            result = run_ocr(
                                temp_path,
                                language=language,
                                side=effective_side,
                                side_source=side_source,
                                template_mapping_confirmed=template_mapping_confirmed and image.size == (1080, 1080),
                                model_dir=model_dir,
                                device="cpu",
                                front_region_profile=try_front_regions,
                                back_region_profile=try_back_regions,
                                engine=get_ocr_engine(language, False),
                                region_engine=get_ocr_engine(language, False) if use_regions else None,
                                # The official PP-OCRv5 baseline never uses the experimental checkpoint's
                                # numeric reversal heuristic.
                                apply_numeric_reading_order_candidate=False,
                            )
                    except Exception as exc:
                        st.error(f"OCR could not run: {exc}")
                        st.info("No fallback or example prediction was substituted. Install/configure PaddleOCR and its inference backend first.")
                    else:
                        result["input_filename"] = uploaded.name
                        with st.container(border=True):
                            preview_col, result_col = st.columns([1, 1.08], gap="large")
                            with preview_col:
                                st.markdown('<div class="section-title" dir="rtl">معاينة الصورة</div>', unsafe_allow_html=True)
                                st.image(image, caption=f"الصورة الأصلية · {image.width}×{image.height}", use_container_width=True)
                                with st.expander("عرض صناديق النص المكتشفة"):
                                    st.image(visualize_boxes(image, result["raw_lines"]), caption="كل الصناديق تحتاج مراجعة", use_container_width=True)
                            with result_col:
                                st.markdown('<div class="section-title" dir="rtl">البيانات المستخرجة</div>', unsafe_allow_html=True)
                                st.caption("النتائج مرشحات OCR للمراجعة، وليست بيانات معتمدة.")
                                if result.get("header_text_suggestion"):
                                    header = result["header_text_suggestion"]
                                    st.info(f"اقتراح لعبارة رأس البطاقة: **{header['suggested_text']}** · يحتاج تأكيدًا بصريًا.")
                                if result.get("fields") is not None:
                                    st.caption("حقول القالب المؤكدة · كل مرشح يحتاج مراجعة بصرية.")
                                    for field_name, field in result["fields"].items():
                                        st.write(f"**{field_name}:** {field['raw_ocr_output'] if field['raw_ocr_output'] is not None else 'null'} · {field['review_status']} · {field['confidence']}")
                                if result.get("region_candidates"):
                                    for field_name, candidate in result["region_candidates"].items():
                                        render_region_candidate(field_name, candidate, image)

                        if result.get("region_candidates"):
                            confidence_items = [(FIELD_LABELS_AR.get(name, name), item.get("confidence")) for name, item in result["region_candidates"].items() if isinstance(item.get("confidence"), (int, float))]
                            if confidence_items:
                                with st.container(border=True):
                                    st.markdown('<div class="section-title" dir="rtl">درجات النموذج لكل حقل</div>', unsafe_allow_html=True)
                                    st.caption("هذه درجات النموذج وليست نسب دقة أو احتمالًا لصحة النص.")
                                    score_cols = st.columns(min(5, len(confidence_items)), gap="small")
                                    for index, (label, score) in enumerate(confidence_items):
                                        normalized = max(0.0, min(1.0, float(score)))
                                        with score_cols[index % len(score_cols)].container(border=True):
                                            st.markdown(f"**{label}**")
                                            st.write(f"{normalized:.1%}")
                                            st.markdown(f'<div class="confidence-track"><div class="confidence-fill" style="width:{normalized * 100:.1f}%"></div></div>', unsafe_allow_html=True)
                        if result.get("mixed_numeric_candidates"):
                            with st.expander("سطور أرقام تحتاج إسنادًا للمراجعة"):
                                st.caption("النظام لا يفصل الرقم القومي عن التاريخ من سطر واحد إلا إذا كان الفصل وحيدًا وبأطوال صحيحة.")
                                for candidate in result["mixed_numeric_candidates"]:
                                    st.code(candidate["raw_text"], language=None)
                                    if candidate.get("national_id_candidate"):
                                        st.info(f"اقتراح يحتاج مراجعة — تاريخ: {candidate['suggested_date_prefix']} · رقم قومي: {candidate['national_id_candidate']}")
                                    else:
                                        st.warning(candidate.get("note", "لا يمكن إسناد هذا السطر للتاريخ أو الرقم القومي بأمان."))
                        with st.expander("عرض كل سطور OCR الخام"):
                            line_rows = [{"رقم السطر": index, "النص الخام": line.get("raw_text", ""), "درجة النموذج (ليست دقة)": line.get("confidence"), "الصندوق": json.dumps(line.get("bbox"), ensure_ascii=False)} for index, line in enumerate(result["raw_lines"], 1)]
                            st.dataframe(line_rows, use_container_width=True, hide_index=True)
                        downloads = st.columns(2)
                        downloads[0].download_button("تحميل النتائج JSON", json.dumps(result, ensure_ascii=False, indent=2), file_name="ocr_result.json", mime="application/json", use_container_width=True)
                        downloads[1].download_button("تحميل النتائج CSV", result_csv(result), file_name="ocr_result.csv", mime="text/csv", use_container_width=True)
