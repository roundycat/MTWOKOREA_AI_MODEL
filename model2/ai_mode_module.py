# ai_mode_module.py

import re
import pandas as pd
from collections import defaultdict
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 전역 변수들
vectorizer = None
df_model = None
preprocessor = None
required_cols = ['L1 NAME', 'L2 NAME', 'L3 NAME', 'L4 NAME']
hierarchical_models = {}
hierarchical_label_encoders = {}

# ✅ SimplePreprocessor 클래스 정의
class SimplePreprocessor:
    def __init__(self, df_model: pd.DataFrame):
        self.df_model = df_model
        self.stop_words = set(["OF", "FOR", "WITH", "AND", "THE", "A", "AN", "TO", "IN", "ON", "BY"])
        self.unit_dict = self._learn_units()
        self.synonyms = self._learn_synonyms()
        self.manual_synonyms = {
    "SET": ["SET", "SETS", "KIT", "KITS", "KITAGAWA", "KITGRACO", "LIGHTINGKIT"],
    "BOX": ["BOX", "BOXES", "BOXING", "CASE", "CASES", "CARTON"],
    "BOTTLE": ["BTL", "BTLS", "BOTTLE", "BOTTLES", "FLASK"],
    "SPOON": ["SPOON", "SPOONS", "TEASPOON", "TABLESPOON", "LADLE"],
    "GLASS": ["GLASS", "GLASSES", "GLASSWARE", "FIBERGLASS", "FIBREGLASS", "MARINEGLASS"],
    "JAR": ["JAR", "JARS", "JARLSBERG", "CANISTER", "CONTAINER"],
    "MUG": ["MUG", "MUGS", "CUP", "CUPS", "TUMBLER", "STEIN"],
    "TRAY": ["TRAY", "TRAYS", "PLATE", "PLATES", "DISH", "DISHES"],
    "TAPE": ["TAPE", "TAPES", "BAND", "STRAP", "STRAPS"],
    "HANDLE": ["HANDLE", "KNOB", "GRIP", "KNOBSET", "LEVER"],
    "VALVE": ["VALVE", "VALVES", "COCK", "STOPPER", "TAP"],
    "PIPE": ["PIPE", "TUBE", "TUBES", "HOSE", "DUCT", "CYLINDER"],
    "BLADE": ["BLADE", "BLADES", "KNIFE", "KNIVES", "CUTTER", "CUTTERS"],
    "COVER": ["COVER", "COVERS", "LID", "CAP", "SLEEVE", "SHIELD"],
    "CLOTH": ["CLOTH", "CLOTHES", "FABRIC", "TEXTILE", "RAG", "RAGS"],
    "GLOVE": ["GLOVE", "GLOVES", "MITT", "MITTS", "HANDGUARD"],
    "WIRE": ["WIRE", "CABLE", "CORD", "ROPE", "LINE"],
    "HELMET": ["HELMET", "CAP", "HEADGEAR", "HAT", "HARDHAT"],
    "GOGGLES": ["GOGGLES", "GLASSES", "SHADES", "VISOR", "PROTECTIVE EYEWEAR"],
    "BATTERY": ["BATTERY", "BATTERIES", "CELL", "ACCUMULATOR"],
    "PAINT": ["PAINT", "COATING", "SPRAY", "VARNISH", "ENAMEL"],
    "BAG": ["BAG", "BAGS", "SACK", "SACKS", "POUCH", "PACK"],
    "LABEL": ["LABEL", "TAG", "STICKER", "DECAL"],
    "CLEANER": ["CLEANER", "DETERGENT", "SOAP", "SANITIZER", "DISINFECTANT"],
    "FAN": ["FAN", "FANS", "VENTILATOR", "BLOWER"],
    "LIGHT": ["LIGHT", "LAMP", "LED", "BULB", "FIXTURE"],
    "PUMP": ["PUMP", "PUMPS", "DISPENSER", "INJECTOR"],
        }
        self.tfidf_model = self._learn_tfidf_model()

    def _learn_units(self):
        units = defaultdict(int)
        specs = self.df_model['L5 NAME (SPEC)'].dropna().astype(str)
        for spec in specs:
            for match in re.findall(r'(\d+)\s*([A-Z]{1,5})', spec.upper()):
                units[match[1]] += 1
        return dict(units)

    def _learn_synonyms(self):
        synonyms = {}
        for col in ['L1 NAME', 'L2 NAME', 'L3 NAME', 'L4 NAME']:
            if col in self.df_model.columns:
                for val in self.df_model[col].dropna().unique():
                    if '/' in val:
                        terms = [t.strip().upper() for t in val.split('/')]
                        for t in terms:
                            synonyms[t] = terms
        return synonyms

    def _learn_tfidf_model(self):
        texts = self.df_model[['L1 NAME', 'L2 NAME', 'L3 NAME', 'L4 NAME']].fillna('').agg(' '.join, axis=1)
        vectorizer = TfidfVectorizer()
        tfidf_model_matrix = vectorizer.fit_transform(texts)
        return dict(zip(vectorizer.get_feature_names_out(), tfidf_model_matrix.sum(axis=0).A1))

    def normalize_query(self, query: str):
        query = query.upper()
        query = re.sub(r"[^A-Z0-9\s/]", " ", query)
        for unit in self.unit_dict:
            query = re.sub(rf"(\d+)\s*{unit}", rf"\1{unit}", query)
        return query.strip()

    def expand_query(self, query: str):
        words = set()
        for token in query.split():
            if token in self.stop_words:
                continue
            words.add(token)
            if token in self.synonyms:
                words.update(self.synonyms[token])
            if token in self.manual_synonyms:
                words.update(self.manual_synonyms[token])
            if '/' in token:
                parts = token.split('/')
                words.update(parts)
                words.add(' '.join(parts))
        return ' '.join(sorted(words))

    def score_query(self, query: str):
        words = query.split()
        return sum(self.tfidf_model.get(w.lower(), 0) for w in words)

    def preprocess(self, query: str):
        norm = self.normalize_query(query)
        expanded = self.expand_query(norm)
        score = self.score_query(expanded)
        return {
            'original': query,
            'normalized': norm,
            'expanded': expanded,
            'tfidf_model_score': round(score, 4)
        }

    def preprocess_normalized(self, query: str):
        return self.normalize_query(query)

# 🔎 예측 함수
def predict_item_name(raw_name, verbose=True):
    raw_name = raw_name.upper()
    normalized = preprocessor.preprocess_normalized(raw_name)

    context = normalized
    prediction = {}

    for col in required_cols:
        model, vec = hierarchical_models[col]
        input_vec = vec.transform([context])
        label_id = model.predict(input_vec)[0]
        label = hierarchical_label_encoders[col].inverse_transform([label_id])[0]
        prediction[col] = label
        context = f"{context} {label}"

    input_vec = vectorizer.transform([normalized])
    candidate_vecs = vectorizer.transform(df_model["ITEM_NAME"])
    similarities = cosine_similarity(input_vec, candidate_vecs).flatten()

    top_idx = similarities.argmax()
    predicted_l5 = df_model.iloc[top_idx]["L5 NAME (SPEC)"]
    prediction["L5 NAME (SPEC)"] = predicted_l5

    if verbose:
        print(f"🔍 L5는 유사도 기반 추천됨 (유사도: {round(similarities[top_idx], 3)})")
    return prediction

from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import numpy as np
import pandas as pd

# 🔧 오타 보정 + 후보 선택 + 유사도 점수 표시
def autocorrect_input(user_input, top_n=3, cutoff=0.6):
    input_upper = user_input.upper()
    candidates = df_model["ITEM_NAME"].unique()

    # 1차 후보: 문자 기반 유사 품명
    close = get_close_matches(input_upper, candidates, n=top_n, cutoff=cutoff)
    if not close:
        return input_upper  # 후보 없음 → 원본 유지

    # TF-Idf_model 유사도 계산
    preprocessed_input = preprocessor.preprocess(input_upper)["normalized"]
    input_vec = vectorizer.transform([preprocessed_input])
    candidate_vecs = vectorizer.transform(close)
    similarities = cosine_similarity(input_vec, candidate_vecs).flatten()

    # 후보 리스트 정리
    scored_candidates = list(zip(close, similarities))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    print("\n🔍 입력하신 품명과 유사한 후보를 찾았어요:")
    for i, (text, score) in enumerate(scored_candidates, 1):
        print(f"  {i}. {text}  (유사도: {round(score, 3)})")

    # 사용자 선택
    try:
        selection = int(input("👉 어떤 후보가 맞나요? (번호 입력, 0 = 원래 입력 유지): "))
        if selection in range(1, len(scored_candidates) + 1):
            return scored_candidates[selection - 1][0]
        else:
            return input_upper
    except ValueError:
        return input_upper


# 🔍 유사 품목 추천
def recommend_similar_items(input_name, top_n=5):
    preprocessed = preprocessor.preprocess(input_name.upper())["normalized"]
    input_vec = vectorizer.transform([preprocessed])

    # L1~L4 구조가 동일한 것만 비교 대상으로 제한
    predicted = predict_item_name(input_name)
    l1 = predicted["L1 NAME"]
    l2 = predicted["L2 NAME"]
    l3 = predicted["L3 NAME"]
    l4 = predicted["L4 NAME"]

    # 후보 필터링
    filtered_df_model = df_model[
        (df_model["L1 NAME"] == l1) &
        (df_model["L2 NAME"] == l2) &
        (df_model["L3 NAME"] == l3) &
        (df_model["L4 NAME"] == l4)
    ]

    if filtered_df_model.empty:
        return []

    candidate_vecs = vectorizer.transform(filtered_df_model["SEARCH_NAME"])
    similarities = cosine_similarity(input_vec, candidate_vecs).flatten()

    top_indices = similarities.argsort()[::-1]
    recommendations = []

    for idx in top_indices:
        if similarities[idx] <= 0:
            continue

        row = filtered_df_model.iloc[idx]
        rec = {
            "SIMILAR_ITEM_NAME": row["ITEM_NAME"],
            "SIMILARITY_SCORE": round(similarities[idx], 3),
            "L1 NAME": row["L1 NAME"],
            "L2 NAME": row["L2 NAME"],
            "L3 NAME": row["L3 NAME"],
            "L4 NAME": row["L4 NAME"],
            "L5 NAME (SPEC)": row["L5 NAME (SPEC)"],
            "P CODE": row["P CODE"]
        }
        recommendations.append(rec)
        if len(recommendations) >= top_n:
            break

    return recommendations


def generate_structured_pcode_based_on_similar(similar_items, fallback_predicted):
    if not similar_items:
        return generate_structured_pcode(fallback_predicted)
    
    # 가장 유사한 기존 품목의 P CODE 접두부 (앞 8자리)
    closest_code = similar_items[0]['P CODE']
    prefix = closest_code[:8]

    # 중복 방지
    existing_codes = df_model["P CODE"].dropna().astype(str)
    used_suffixes = [
        int(code[-3:]) for code in existing_codes
        if code.startswith(prefix) and code[-3:].isdigit()
    ]

    next_suffix = max(used_suffixes, default=0) + 1
    return f"{prefix}{str(next_suffix).zfill(3)}"

def generate_structured_pcode(predicted):
    # 정확한 키 이름으로 추출
    l1 = predicted.get("L1 NAME", "")[:2].upper().ljust(2, "X")
    l2 = predicted.get("L2 NAME", "")[:2].upper().ljust(2, "X")
    l3 = predicted.get("L3 NAME", "")[:2].upper().ljust(2, "X")
    l4 = predicted.get("L4 NAME", "")[:2].upper().ljust(2, "X")

    prefix = f"{l1}{l2}{l3}{l4}"

    existing_codes = df_model["P CODE"].dropna().astype(str)
    used_suffixes = [
        int(code[-3:]) for code in existing_codes
        if code.startswith(prefix) and code[-3:].isdigit()
    ]

    next_suffix = max(used_suffixes, default=0) + 1
    new_pcode = f"{prefix}{str(next_suffix).zfill(3)}"

    return new_pcode

def recommend_similar_pcodes_detailed(new_pcode, top_n=5):
    existing = df_model.dropna(subset=["P CODE"])
    existing_codes = existing["P CODE"].astype(str).unique()

    # 문자 유사도 기반 후보 추출
    similar_codes = get_close_matches(new_pcode, existing_codes, n=top_n * 5, cutoff=0.5)  # 더 많이 뽑고 필터링

    # 기준 분류 추출 (new_pcode 기준)
    base_row = {
        "L1": new_pcode[0:2],
        "L2": new_pcode[2:4],
        "L3": new_pcode[4:6],
        "L4": new_pcode[6:8],
    }

    # 상세 정보 구성 (분류 구조가 동일한 것만)
    results = []
    for code in similar_codes:
        row = existing[existing["P CODE"] == code].iloc[0]
        if (
            row.get("L1 NAME", "")[:2].upper() == base_row["L1"] and
            row.get("L2 NAME", "")[:2].upper() == base_row["L2"] and
            row.get("L3 NAME", "")[:2].upper() == base_row["L3"] and
            row.get("L4 NAME", "")[:2].upper() == base_row["L4"]
        ):
            results.append({
                "P CODE": code,
                "ITEM_NAME": row.get("ITEM_NAME") or row.get("L5 NAME (SPEC)", "N/A"),
                "L1": row.get("L1 NAME", ""),
                "L2": row.get("L2 NAME", ""),
                "L3": row.get("L3 NAME", ""),
                "L4": row.get("L4 NAME", ""),
                "L5": row.get("L5 NAME (SPEC)", "")
            })
        if len(results) >= top_n:
            break

    return pd.DataFrame(results)

# 🔹 여기가 함수 바깥입니다 (맨 왼쪽)

def is_valid_combination(predicted: dict, df_model_ref: pd.DataFrame) -> bool:
    l1, l2, l3, l4, l5 = (
        predicted["L1 NAME"],
        predicted["L2 NAME"],
        predicted["L3 NAME"],
        predicted["L4 NAME"],
        predicted["L5 NAME (SPEC)"],
    )
    return not df_model_ref[
        (df_model_ref["L1 NAME"] == l1)
        & (df_model_ref["L2 NAME"] == l2)
        & (df_model_ref["L3 NAME"] == l3)
        & (df_model_ref["L4 NAME"] == l4)
        & (df_model_ref["L5 NAME (SPEC)"] == l5)
    ].empty


def suggest_similar_combination(predicted: dict, df_model_ref: pd.DataFrame, top_n=5):
    query = " ".join([
        predicted.get("L1 NAME", ""),
        predicted.get("L2 NAME", ""),
        predicted.get("L3 NAME", ""),
        predicted.get("L4 NAME", ""),
        predicted.get("L5 NAME (SPEC)", "")
    ])
    input_vec = vectorizer.transform([preprocessor.preprocess_normalized(query)])
    candidate_vecs = vectorizer.transform(df_model_ref["SEARCH_NAME"])
    similarities = cosine_similarity(input_vec, candidate_vecs).flatten()
    top_indices = similarities.argsort()[::-1]
    results = []
    for idx in top_indices:
        row = df_model_ref.iloc[idx]
        results.append({
            "P CODE": row["P CODE"],
            "SEARCH_NAME": row["SEARCH_NAME"],
            "SIMILARITY_SCORE": round(similarities[idx], 3),
        })
        if len(results) >= top_n:
            break
    return pd.DataFrame(results)

def fallback_by_l1_l3(predicted, input_name, top_n=5, similarity_threshold=0.3):
    l1 = predicted.get("L1 NAME", "")
    l2 = predicted.get("L2 NAME", "")
    l3 = predicted.get("L3 NAME", "")

    # 후보군 필터링
    candidates = df_model[
        (df_model["L1 NAME"] == l1) &
        (df_model["L2 NAME"] == l2) &
        (df_model["L3 NAME"] == l3)
    ]

    if candidates.empty:
        return []

    # 입력 벡터 (SEARCH_NAME 기준 전처리)
    input_query = preprocessor.preprocess_normalized(input_name)
    input_vec = vectorizer.transform([input_query])
    candidate_vecs = vectorizer.transform(candidates["SEARCH_NAME"])

    similarities = cosine_similarity(input_vec, candidate_vecs).flatten()
    top_indices = similarities.argsort()[::-1]

    results = []
    for idx in top_indices:
        sim = similarities[idx]
        if sim < similarity_threshold:
            continue  # 의미 없는 추천은 제외

        row = candidates.iloc[idx]
        results.append({
            "SIMILAR_ITEM_NAME": row["SEARCH_NAME"],
            "SIMILARITY_SCORE": round(sim, 3),
            "P CODE": row["P CODE"],
            "L5 NAME (SPEC)": row["L5 NAME (SPEC)"]
        })

        if len(results) >= top_n:
            break

    return results

def recommend_by_similar_l1_l2_l3(input_name, predicted, top_n=5):
    l1, l2, l3 = predicted["L1 NAME"], predicted["L2 NAME"], predicted["L3 NAME"]

    # 분류 조건에 맞는 후보 추출
    candidates = df_model[
        (df_model["L1 NAME"] == l1) &
        (df_model["L2 NAME"] == l2) &
        (df_model["L3 NAME"] == l3)
    ]

    if candidates.empty:
        return []

    input_vec = vectorizer.transform([preprocessor.preprocess(input_name)["normalized"]])
    candidate_vecs = vectorizer.transform(candidates["SEARCH_NAME"])
    similarities = cosine_similarity(input_vec, candidate_vecs).flatten()
    top_indices = similarities.argsort()[::-1]

    results = []
    for idx in top_indices:
        row = candidates.iloc[idx]
        results.append({
            "SIMILAR_ITEM_NAME": row["SEARCH_NAME"],
            "SIMILARITY_SCORE": round(similarities[idx], 3),
            "P CODE": row["P CODE"],
            "L5 NAME (SPEC)": row["L5 NAME (SPEC)"]
        })
        if len(results) >= top_n:
            break

    return results

def recommend_global_similar_items(input_name, top_n=5):
    query = preprocessor.preprocess(input_name)["normalized"]
    input_vec = vectorizer.transform([query])
    candidate_vecs = vectorizer.transform(df_model["SEARCH_NAME"])

    similarities = cosine_similarity(input_vec, candidate_vecs).flatten()
    top_indices = similarities.argsort()[::-1]

    results = []
    for idx in top_indices:
        row = df_model.iloc[idx]
        results.append({
            "SIMILAR_ITEM_NAME": row["SEARCH_NAME"],
            "SIMILARITY_SCORE": round(similarities[idx], 3),
            "P CODE": row["P CODE"],
            "L1 NAME": row["L1 NAME"],
            "L2 NAME": row["L2 NAME"],
            "L3 NAME": row["L3 NAME"],
            "L4 NAME": row["L4 NAME"],
            "L5 NAME (SPEC)": row["L5 NAME (SPEC)"],
        })
        if len(results) >= top_n:
            break

    return results

def keyword_fallback_items(input_text, top_n=5):
    words = input_text.upper().split()

    # Step 1: 모든 단어 포함 (AND)
    matches_all = df_model.copy()
    for word in words:
        matches_all = matches_all[matches_all["SEARCH_NAME"].str.contains(word, na=False)]

    if not matches_all.empty:
        return format_keyword_results(matches_all, top_n, mode="AND")

    # Step 2: 일부 단어 포함 (OR)
    pattern = "|".join(words)
    matches_any = df_model[df_model["SEARCH_NAME"].str.contains(pattern, na=False)]

    if not matches_any.empty:
        return format_keyword_results(matches_any, top_n, mode="OR")

    return []

def format_keyword_results(df_model_matches, top_n=5, mode=""):
    results = []
    for _, row in df_model_matches.head(top_n).iterrows():
        results.append({
            "P CODE": row["P CODE"],
            "SEARCH_NAME": row["SEARCH_NAME"],
            "L1 NAME": row["L1 NAME"],
            "L2 NAME": row["L2 NAME"],
            "L3 NAME": row["L3 NAME"],
            "L4 NAME": row["L4 NAME"],
            "L5 NAME (SPEC)": row["L5 NAME (SPEC)"],
            "MATCH_MODE": mode
        })
    return results

def extract_main_keyword(text, common_words=["ELECTRIC", "PRODUCT", "SET"]):
    words = text.upper().split()
    keywords = [w for w in words if w not in common_words]
    return keywords[0] if keywords else text

# ✅ 대화형 예측 함수
def interactive_product_recommendation():
    user_input = input("🔍 품명을 입력하세요: ").strip()
    if not user_input:
        print("❗ 입력이 비어 있습니다.")
        return

    # 1. 오타 보정
    corrected = autocorrect_input(user_input)
    if corrected != user_input.upper():
        print(f"🔧 오타 보정: '{user_input}' → '{corrected}'")
    else:
        print(f"✅ 입력 인식됨: '{corrected}'")

    # 2. 품명 예측
    predicted = predict_item_name(corrected)

    # 3. 예측 신뢰도 평가 함수
    def is_prediction_reasonable(predicted, user_input, threshold=0.3):
        pred_string = f"{predicted.get('L1 NAME', '')} {predicted.get('L2 NAME', '')} {predicted.get('L3 NAME', '')} {predicted.get('L4 NAME', '')} {predicted.get('L5 NAME (SPEC)', '')}"
        user_vec = vectorizer.transform([user_input.upper()])
        pred_vec = vectorizer.transform([pred_string.upper()])
        sim = cosine_similarity(user_vec, pred_vec)[0][0]
        if len(user_input.strip().split()) <= 1:
            print("⚠️ 입력이 너무 짧습니다. 신뢰도 판단 없이 fallback 진행될 수 있습니다.")
            return False
        print(f"📊 예측 신뢰도 (입력 vs 예측): {round(sim * 100, 1)}%")
        return sim >= threshold

    # 4. 전역 유사 품목 추천
    global_recommendations = recommend_global_similar_items(corrected)
    if global_recommendations and global_recommendations[0]["SIMILARITY_SCORE"] >= 0.4:
        print("\n🔎 전역 유사 품목 Top 5:")
        for rec in global_recommendations:
            print(f"[{rec['SIMILARITY_SCORE']}] {rec['SIMILAR_ITEM_NAME']} → {rec['P CODE']}")
    else:
        print("📭 전역 유사 품목 부족. fallback을 시도합니다.")

    # 5. 예측 신뢰도 낮으면 fallback
    if not is_prediction_reasonable(predicted, corrected):
        print("⚠️ 예측된 분류가 입력과 너무 다릅니다. fallback 중...")
        similar_items = recommend_similar_items(corrected)

        if similar_items:
            best_fallback = similar_items[0]["SIMILAR_ITEM_NAME"]
            predicted = predict_item_name(best_fallback, verbose=False)
            print(f"🔁 fallback 예측된 품명: '{best_fallback}' → 재예측됨")
        else:
            print("📭 유사한 품명이 없음. L1~L3 기반 fallback 시도 중...")
            l3_fallbacks = fallback_by_l1_l3(predicted, corrected)
            if l3_fallbacks:
                best_fallback = l3_fallbacks[0]["SIMILAR_ITEM_NAME"]
                predicted = predict_item_name(best_fallback, verbose=False)
                print(f"🔁 L1~L3 기반 fallback 예측된 품명: '{best_fallback}' → 재예측됨")
            else:
                print("❌ 모든 fallback 실패")

    # 6. 최종 예측 분류 출력
    print("\n🎯 예측된 분류:")
    for k, v in predicted.items():
        print(f"- {k}: {v}")

    # 7. 정합성 체크
    if not is_valid_combination(predicted, df_model):
        print("❌ 예측된 L1~L5 조합이 실제 존재하지 않습니다.")
        print("🔁 유사한 L1~L5 조합 추천:")
        display(suggest_similar_combination(predicted, df_model))

    # 8. 기존 P CODE 확인 또는 신규 생성
    matched = df_model[
        (df_model["L1 NAME"] == predicted["L1 NAME"]) &
        (df_model["L2 NAME"] == predicted["L2 NAME"]) &
        (df_model["L3 NAME"] == predicted["L3 NAME"]) &
        (df_model["L4 NAME"] == predicted["L4 NAME"]) &
        (df_model["L5 NAME (SPEC)"] == predicted["L5 NAME (SPEC)"])
    ]

    if not matched.empty:
        existing_pcode = matched["P CODE"].iloc[0]
        print(f"- 📦 추천 P CODE (기존): {existing_pcode}")
    else:
        # global_recommendations 우선 사용, 없으면 similar_items 대체
        base_items = global_recommendations or recommend_similar_items(corrected)
        new_pcode = generate_structured_pcode_based_on_similar(base_items, predicted)
        print(f"- 🆕 추천 P CODE (신규): {new_pcode}")
        similar_details_df_model = recommend_similar_pcodes_detailed(new_pcode)
        if not similar_details_df_model.empty:
            print("\n📋 유사한 기존 P CODE 및 품명:")
            display(similar_details_df_model)
        else:
            print("📭 유사한 기존 P CODE 없음")

    # 9. TF-Idf_model 유사 품목 추천
    recommendations = recommend_similar_items(corrected)
    if recommendations and any(rec['SIMILARITY_SCORE'] > 0 for rec in recommendations):
        print("\n🔎 유사한 품목 Top 5:")
        for rec in recommendations:
            print(f"[{rec['SIMILARITY_SCORE']}] {rec['SIMILAR_ITEM_NAME']} → {rec['P CODE']}")
    else:
        print("📭 TF-Idf_model 기반 추천 실패. 전역 검색 시도 중...")
        recommendations = global_recommendations or []
        if not recommendations:
            print("📭 전역 유사 품목도 없음. 키워드 포함 검색 시도 중...")
            keyword_matches = keyword_fallback_items(corrected)
            if keyword_matches:
                print("\n🔎 단어 포함 품목 추천:")
                for item in keyword_matches:
                    print(f"🔹 [{item.get('MATCH_MODE', '키워드')}] {item['SEARCH_NAME']} → {item['P CODE']}")
            else:
                print("❌ 단어 포함 결과도 없음.")


# ai_model_module.py 내부에 추가할 코드
def set_globals(vec, df_model_param, prep, models, encoders, valid_combos=None):
    global vectorizer, df_model, preprocessor
    global hierarchical_models, hierarchical_label_encoders, valid_combinations

    vectorizer = vec
    df_model = df_model_param
    preprocessor = prep
    hierarchical_models = models
    hierarchical_label_encoders = encoders
    valid_combinations = valid_combos or set()

