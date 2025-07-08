import pandas as pd
import numpy as np
import re
from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter
from rapidfuzz import fuzz, process
from googletrans import Translator
import langdetect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss
import warnings
warnings.filterwarnings('ignore')

# ========================= 전처리기 클래스 =========================
class ShipSupplyPreprocessor:
    """선박용품 도메인 특화 전처리기 - 완전 데이터 기반 학습"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.translator = Translator()
        
        print("데이터에서 패턴 학습 중...")
        
        # 데이터에서 모든 패턴 자동 학습
        self.learn_from_data()
        
        print("전처리기 학습 완료")
        
    def learn_from_data(self):
        """데이터에서 모든 패턴을 자동으로 학습"""
        
        # 1. 단위 패턴 학습
        print("  1. 단위 패턴 학습 중...")
        self.unit_patterns = self._learn_unit_patterns()
        
        # 2. 복합어 패턴 학습
        print(" 2. 복합어 패턴 학습 중...")
        self.compound_patterns = self._learn_compound_patterns()
        
        # 3. 동의어 관계 학습
        print(" 3. 동의어 관계 학습 중...")
        self.synonyms = self._learn_synonyms()
        
        # 4. 오타 보정을 위한 어휘 빈도 학습
        print(" 4. 어휘 빈도 학습 중...")
        self.vocabulary_freq = self._learn_vocabulary_frequency()
        
        # 5. 카테고리 구조 학습
        print(" 5. 카테고리 구조 학습 중...")
        self.category_structure = self._learn_category_structure()
        
        # 6. 문맥 기반 단어 관계 학습
        print(" 6. 문맥 정보 학습 중...")
        self.word_context = self._learn_word_context()
    
    def _learn_unit_patterns(self) -> Dict[str, str]:
        """데이터에서 단위 패턴 자동 학습"""
        unit_patterns = {}
        
        # L5 (규격) 컬럼에서 패턴 추출
        if 'L5 NAME (SPEC)' in self.df.columns:
            specs = self.df['L5 NAME (SPEC)'].dropna().astype(str)
            
            # 숫자+단위 패턴 찾기
            pattern_counts = Counter()
            
            for spec in specs:
                # 숫자와 단위 찾기 (예: 750ML, 1L, 12BTL)
                matches = re.findall(r'(\d+)\s*([A-Z]+)', spec)
                for num, unit in matches:
                    pattern_counts[unit] += 1
            
            # 빈도수 높은 단위들의 변형 학습
            common_units = [unit for unit, count in pattern_counts.items() if count > 5]
            
            for unit in common_units:
                # 같은 의미로 보이는 변형들 찾기
                variations = []
                for spec in specs:
                    # ML, MLS, MLTS 같은 변형 찾기
                    if unit in spec:
                        context = re.findall(rf'\d+\s*([A-Z]*{unit}[A-Z]*)', spec)
                        variations.extend(context)
                
                # 가장 짧은 형태를 표준으로
                if variations:
                    variation_set = set(variations)
                    if len(variation_set) > 1:
                        standard = min(variation_set, key=len)
                        for var in variation_set:
                            if var != standard and len(var) <= len(standard) + 3:
                                pattern = rf'\b(\d+)\s*{re.escape(var)}\b'
                                replacement = rf'\1{standard}'
                                unit_patterns[pattern] = replacement
        
        # 연도 패턴 (12Y → 12YEAR)
        year_pattern = r'\b(\d+)\s*Y\b'
        if any(re.search(year_pattern, str(text)) for text in self.df.values.flatten() if pd.notna(text)):
            unit_patterns[year_pattern] = r'\1YEAR'
        
        return unit_patterns
    
    def _learn_compound_patterns(self) -> Dict[str, List[str]]:
        """데이터에서 복합어 패턴 자동 학습"""
        compound_patterns = defaultdict(list)
        
        # 모든 카테고리명에서 복합어 패턴 추출
        all_names = []
        for col in ['L1 NAME', 'L2 NAME', 'L3 NAME', 'L4 NAME']:
            if col in self.df.columns:
                all_names.extend(self.df[col].dropna().astype(str).tolist())
        
        # 빈도 계산을 위한 카운터
        name_counter = Counter(all_names)
        
        # 2개 이상의 단어로 구성된 용어 찾기
        for name in set(all_names):
            words = name.split()
            if len(words) >= 2:
                # 이 복합어가 데이터에 얼마나 자주 나타나는지
                compound_count = name_counter[name]
                
                # 각 단어가 개별적으로 나타나는 빈도
                individual_counts = []
                for word in words:
                    word_count = sum(1 for n in all_names if word in n.split())
                    individual_counts.append(word_count)
                
                # 복합어로 더 자주 나타나면 복합어로 처리
                # (복합어 빈도가 개별 단어 최소 빈도의 30% 이상)
                if compound_count > min(individual_counts) * 0.3:
                    compound_patterns[name] = words
        
        return dict(compound_patterns)
    
    def _learn_synonyms(self) -> Dict[str, Set[str]]:
        """데이터에서 동의어 관계 자동 학습"""
        synonyms = defaultdict(set)
        
        # 1. 슬래시로 구분된 동의어
        for col in ['L1 NAME', 'L2 NAME', 'L3 NAME', 'L4 NAME']:
            if col in self.df.columns:
                for name in self.df[col].dropna().unique():
                    if '/' in str(name):
                        parts = [p.strip() for p in str(name).split('/')]
                        for part in parts:
                            synonyms[part].update(parts)
        
        # 2. 같은 상위 카테고리를 공유하는 유사 단어들
        category_words = defaultdict(lambda: defaultdict(int))
        
        for _, row in self.df.iterrows():
            if pd.notna(row.get('L1 NAME')) and pd.notna(row.get('L2 NAME')):
                l1 = str(row['L1 NAME'])
                l2_words = str(row['L2 NAME']).split()
                
                for word in l2_words:
                    if len(word) > 3:  # 의미있는 길이의 단어만
                        category_words[l1][word] += 1
        
        # 같은 카테고리에서 자주 나타나는 단어들을 동의어로
        for category, words in category_words.items():
            word_list = [w for w, count in words.items() if count > 5]
            
            # 편집 거리가 가까운 단어들을 동의어로
            for i, word1 in enumerate(word_list):
                for word2 in word_list[i+1:]:
                    similarity = fuzz.ratio(word1, word2)
                    if similarity > 80:  # 80% 이상 유사
                        synonyms[word1].add(word2)
                        synonyms[word2].add(word1)
        
        # 3. 일반적인 동의어 패턴 (CIGARETTE/CIGARETE 같은)
        all_words = set()
        for col in self.df.columns:
            if 'NAME' in col:
                words = self.df[col].dropna().astype(str).str.split().explode()
                all_words.update(words)
        
        # 편집거리 1-2인 단어들 중 하나가 훨씬 빈번하면 동의어로
        word_freq = Counter()
        for word in all_words:
            word_freq[word] += 1
        
        for word1, freq1 in word_freq.most_common(500):
            for word2, freq2 in word_freq.items():
                if word1 != word2 and 1 <= len(word1) - len(word2) <= 2:
                    if fuzz.ratio(word1, word2) > 85 and freq1 > freq2 * 5:
                        synonyms[word2].add(word1)
        
        return dict(synonyms)
    
    def _learn_vocabulary_frequency(self) -> Dict[str, int]:
        """어휘 빈도 학습 (오타 보정용)"""
        word_freq = Counter()
        
        # 모든 텍스트에서 단어 빈도 계산
        for col in self.df.columns:
            if 'NAME' in col or 'CODE' in col:
                for text in self.df[col].dropna().astype(str):
                    words = text.split()
                    word_freq.update(words)
        
        return dict(word_freq)
    
    def _learn_category_structure(self) -> Dict:
        """카테고리 계층 구조 학습"""
        structure = {
            'hierarchy': defaultdict(lambda: defaultdict(set)),
            'level_words': defaultdict(set),
            'level_importance': {}  # 각 레벨의 중요도
        }
        
        # 계층 관계 학습
        for _, row in self.df.iterrows():
            for i, level in enumerate(['L1 NAME', 'L2 NAME', 'L3 NAME', 'L4 NAME']):
                if level in row and pd.notna(row[level]):
                    current = str(row[level])
                    
                    # 현재 레벨의 단어들 저장
                    words = current.split()
                    structure['level_words'][level].update(words)
                    
                    # 상위 레벨과의 관계 저장
                    if i > 0:
                        prev_level = ['L1 NAME', 'L2 NAME', 'L3 NAME', 'L4 NAME'][i-1]
                        if prev_level in row and pd.notna(row[prev_level]):
                            parent = str(row[prev_level])
                            structure['hierarchy'][parent][level].add(current)
        
        # 레벨별 중요도 계산 (상위 레벨일수록 중요)
        level_weights = {'L1 NAME': 4, 'L2 NAME': 3, 'L3 NAME': 2, 'L4 NAME': 1, 'L5 NAME (SPEC)': 0.5}
        structure['level_importance'] = level_weights
        
        return structure
    
    def _learn_word_context(self) -> Dict:
        """단어의 문맥 정보 학습 (TF-IDF 기반)"""
        # 모든 텍스트 수집
        texts = []
        text_levels = []  # 텍스트가 어느 레벨에서 왔는지
        
        for col in ['L1 NAME', 'L2 NAME', 'L3 NAME', 'L4 NAME']:
            if col in self.df.columns:
                col_texts = self.df[col].dropna().astype(str).tolist()
                texts.extend(col_texts)
                text_levels.extend([col] * len(col_texts))
        
        if not texts:
            return {}
        
        # TF-IDF 벡터화
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # unigram + bigram
            max_features=1000,
            min_df=2,
            max_df=0.8
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # 단어별 중요도 점수
            word_importance = {}
            for idx, word in enumerate(feature_names):
                importance = tfidf_matrix[:, idx].sum()
                word_importance[word] = float(importance)
            
            return {
                'vectorizer': vectorizer,
                'tfidf_matrix': tfidf_matrix,
                'word_importance': word_importance,
                'texts': texts,
                'text_levels': text_levels
            }
        except:
            return {}
    
    def normalize_text(self, text: str) -> str:
        """학습된 패턴 기반 텍스트 정규화"""
        if pd.isna(text):
            return ""
    
        text = str(text).upper().strip()
    
        # 1. 학습된 단위 패턴 적용
        for pattern, replacement in self.unit_patterns.items():
            # 1X12 같은 패턴 보호
            if re.search(r'\d+X\d+', text):  # 텍스트에 숫자X숫자가 있으면
                continue  # 이 패턴은 건너뛰기
        
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
        # 2. 특수문자 정리 (슬래시, 하이픈, &는 유지)
        text = re.sub(r'[^\w\s/\-&]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
    
        return text
    
    def smart_correct(self, query: str, threshold: float = 0.8) -> str:
        """학습된 어휘 기반 스마트 보정"""
        words = query.split()
        corrected_words = []
        
        
        for word in words:
            # 빈도수가 높은 단어는 그대로 유지
            if word in self.vocabulary_freq and self.vocabulary_freq[word] > 10:
                corrected_words.append(word)
                continue
            
            # 빈도수가 낮거나 없는 단어는 유사 단어 찾기
            candidates = []
            if word.isdigit() or re.search(r'\d', word):
                corrected_words.append(word)
                continue
            # 복합어(언더스코어 포함)는 보정하지 않음
            if '_' in word:
                corrected_words.append(word)
                continue
            
            # 먼저 동의어 확인
            if word in self.synonyms and self.synonyms[word]:
                # 동의어 중 가장 빈번한 것 선택
                best_synonym = max(self.synonyms[word], 
                                 key=lambda x: self.vocabulary_freq.get(x, 0))
                corrected_words.append(best_synonym)
                continue
            
            # 유사 단어 찾기
            for vocab_word, freq in self.vocabulary_freq.items():
                # 길이가 비슷한 단어만 비교 (효율성)
                if abs(len(vocab_word) - len(word)) <= 2:
                    similarity = fuzz.ratio(word, vocab_word) / 100.0
                    if similarity >= threshold:
                        # 빈도수를 가중치로 사용
                        score = similarity * np.log(freq + 1)
                        candidates.append((vocab_word, score))
            
            if candidates:
                # 가장 높은 점수의 단어 선택
                best_word = max(candidates, key=lambda x: x[1])[0]
                corrected_words.append(best_word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def detect_compound_words(self, query: str) -> str:
        """학습된 복합어 패턴 감지 및 처리"""
        # 쿼리에 포함된 복합어 찾기
        query_words = query.split()
        
        for compound, compound_words in sorted(self.compound_patterns.items(), 
                                              key=lambda x: len(x[1]), reverse=True):
            # 모든 구성 단어가 순서대로 있는지 확인
            for i in range(len(query_words) - len(compound_words) + 1):
                if query_words[i:i+len(compound_words)] == compound_words:
                    # 복합어로 대체 (언더스코어로 연결)
                    new_query_words = (query_words[:i] + 
                                     ['_'.join(compound_words)] + 
                                     query_words[i+len(compound_words):])
                    query = ' '.join(new_query_words)
                    query_words = new_query_words
                    break
        
        return query
    
    def translate_if_korean(self, text: str) -> str:
        """한글 감지 및 번역"""
        try:
            if re.search('[가-힣]', text):
                result = self.translator.translate(text, src='ko', dest='en')
                return result.text.upper()
        except:
            pass
        return text
    
    def context_aware_expansion(self, query: str) -> List[str]:
        """문맥 기반 쿼리 확장"""
        expanded = [query]
        
        # 1. 학습된 동의어 확장
        words = query.split()
        for word in words:
            if word in self.synonyms:
                for synonym in self.synonyms[word]:
                    new_query = query.replace(word, synonym)
                    if new_query not in expanded:
                        expanded.append(new_query)
        
        # 2. TF-IDF 기반 유사 문맥 찾기
        if hasattr(self, 'word_context') and 'vectorizer' in self.word_context:
            try:
                query_vec = self.word_context['vectorizer'].transform([query])
                similarities = cosine_similarity(query_vec, self.word_context['tfidf_matrix'])[0]
                
                # 상위 유사 텍스트 찾기
                top_indices = similarities.argsort()[-3:][::-1]
                for idx in top_indices:
                    if similarities[idx] > 0.3:
                        similar_text = self.word_context['texts'][idx]
                        if similar_text not in expanded and len(expanded) < 5:
                            expanded.append(similar_text)
            except:
                pass
        
        return expanded[:5]
    
    def preprocess_query(self, query: str, verbose: bool = False) -> str:
        """쿼리 전처리 메인 함수 - 완전 학습 기반"""
        steps = []
        current = query
        
        # 1. 기본 정규화
        normalized = self.normalize_text(current)
        if normalized != current:
            steps.append(('정규화', normalized))
            current = normalized
        
        # 2. 한글 번역
        translated = self.translate_if_korean(current)
        if translated != current:
            steps.append(('번역', translated))
            current = translated
        
        # 3. 복합어 감지
        compound_detected = self.detect_compound_words(current)
        if compound_detected != current:
            steps.append(('복합어', compound_detected))
            current = compound_detected
        
        # 4. 스마트 보정
        corrected = self.smart_correct(current)
        if corrected != current:
            steps.append(('보정', corrected))
            current = corrected
        
        # 5. 복합어 원복 (언더스코어 제거)
        final = current.replace('_', ' ')
        if final != current:
            steps.append(('최종', final))
        
        if verbose:
            print(f"전처리 과정: {query}")
            for step_name, result in steps:
                print(f"   → {step_name}: {result}")
            if not steps:
                print(f"   → 변경 없음: {query}")
        
        return final
    
    def create_search_text(self, row: pd.Series) -> str:
        """검색용 텍스트 생성 - 학습된 중요도 기반"""
        parts = []
        weights = []
        
        # 각 레벨의 텍스트와 중요도 계산
        for level in ['L1 NAME', 'L2 NAME', 'L3 NAME', 'L4 NAME', 'L5 NAME (SPEC)']:
            if level in row and pd.notna(row[level]) and row[level]:
                text = str(row[level])
                parts.append(text)
                
                # 레벨별 가중치 적용
                if hasattr(self, 'category_structure'):
                    level_weight = self.category_structure['level_importance'].get(level, 1)
                    
                    # 단어 중요도도 고려
                    if hasattr(self, 'word_context') and 'word_importance' in self.word_context:
                        words = text.lower().split()
                        word_importance = sum(
                            self.word_context['word_importance'].get(word, 0) 
                            for word in words
                        ) / len(words) if words else 0
                        
                        # 최종 가중치 = 레벨 가중치 × 단어 중요도
                        final_weight = level_weight * (1 + word_importance)
                    else:
                        final_weight = level_weight
                    
                    # 가중치에 따라 반복
                    repeat_count = max(1, int(final_weight))
                    weights.extend([text] * repeat_count)
        
        # P CODE 추가
        if 'P CODE' in row and pd.notna(row['P CODE']):
            parts.append(str(row['P CODE']))
        
        # 조합
        combined = ' | '.join(parts)
        if weights:
            weighted_text = ' '.join(weights)
            combined = f"{combined} {weighted_text}"
        
        return self.normalize_text(combined)
    
    def get_preprocessing_stats(self) -> Dict:
        """학습된 전처리 통계 반환"""
        stats = {
            '학습된 단위 패턴': len(self.unit_patterns),
            '학습된 복합어': len(self.compound_patterns),
            '학습된 동의어 그룹': len(self.synonyms),
            '학습된 어휘 수': len(self.vocabulary_freq),
            '가장 빈번한 단어 Top10': Counter(self.vocabulary_freq).most_common(10)
        }
        
        # 학습된 패턴 예시
        if self.unit_patterns:
            stats['단위 패턴 예시'] = list(self.unit_patterns.items())[:5]
        if self.compound_patterns:
            stats['복합어 예시'] = list(self.compound_patterns.items())[:5]
        if self.synonyms:
            stats['동의어 예시'] = [(k, list(v)[:3]) for k, v in 
                                  list(self.synonyms.items())[:5]]
        
        return stats


# ========================= 임베딩 검색 엔진 =========================
class EmbeddingSearchEngine:
    """임베딩 기반 검색 엔진"""
    
    def __init__(self, model_name='paraphrase-multilingual-mpnet-base-v2'):
        print(f"임베딩 모델 로딩: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.df = None
        print("모델 로드 완료")
        
    def build_index(self, df, preprocessor, sample_size=None):
        """FAISS 인덱스 구축"""
        self.df = df
        
        # 샘플링 (테스트용)
        if sample_size:
            df_sample = df.sample(min(sample_size, len(df)))
            print(f"샘플 {len(df_sample)}개로 인덱스 구축")
        else:
            df_sample = df
        
        print("검색용 텍스트 생성 중...")
        
        # 검색용 텍스트 생성
        texts = []
        for idx, row in df_sample.iterrows():
            search_text = preprocessor.create_search_text(row)
            texts.append(search_text)
        
        print(f"{len(texts)}개 텍스트 임베딩 중...")
        
        # 배치로 임베딩 생성
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        # FAISS 인덱스 생성 - 코사인 유사도용으로 변경
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # L2 대신 IP(Inner Product) 사용
        
        # 벡터 정규화 (코사인 유사도를 위해)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        print(f"FAISS 인덱스 구축 완료: {len(embeddings)}개 벡터")
        
    def search(self, query, k=5):
        """유사도 검색"""
        # 쿼리 임베딩
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )
        
        # 쿼리 벡터 정규화
        faiss.normalize_L2(query_embedding)
        
        # 검색
        similarities, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                continue
            
            row = self.df.iloc[idx]
            
            # Inner Product 결과는 이미 0~1 범위의 코사인 유사도
            confidence = float(sim)
            
            results.append({
                'code': row['P CODE'],
                'confidence': confidence,
                'distance': float(sim),  # 호환성을 위해 distance로도 저장
                'path': ' > '.join([
                    str(row[col]) for col in ['L1 NAME', 'L2 NAME', 'L3 NAME', 'L4 NAME']
                    if pd.notna(row[col]) and row[col]
                ]),
                'spec': row.get('L5 NAME (SPEC)', ''),
                'full_info': row.to_dict()
            })
        
        return results


# ========================= 카테고리 분류기 =========================
class CategoryClassifier:
    """카테고리 분류기 (간단한 규칙 기반)"""
    
    def __init__(self, df):
        self.df = df
        self.build_category_map()
        print("분류기 초기화 완료")
        
    def build_category_map(self):
        """카테고리별 키워드 매핑 구축"""
        self.keyword_to_category = {}
        
        # 각 레벨별로 키워드 추출
        for level in ['L1 NAME', 'L2 NAME', 'L3 NAME']:
            if level not in self.df.columns:
                continue
                
            for category in self.df[level].dropna().unique():
                # 카테고리명의 각 단어를 키워드로
                words = str(category).split()
                for word in words:
                    if len(word) > 2:  # 너무 짧은 단어 제외
                        word = word.upper()
                        if word not in self.keyword_to_category:
                            self.keyword_to_category[word] = []
                        self.keyword_to_category[word].append({
                            'category': category,
                            'level': level
                        })
    
    def classify(self, query):
        """쿼리를 카테고리로 분류"""
        query_words = query.upper().split()
        category_scores = {}
        
        # 각 단어별로 매칭되는 카테고리 찾기
        for word in query_words:
            if word in self.keyword_to_category:
                for cat_info in self.keyword_to_category[word]:
                    cat_key = f"{cat_info['level']}:{cat_info['category']}"
                    if cat_key not in category_scores:
                        category_scores[cat_key] = 0
                    category_scores[cat_key] += 1
        
        if not category_scores:
            return None
        
        # 가장 높은 점수의 카테고리
        best_category = max(category_scores, key=category_scores.get)
        level, category = best_category.split(':', 1)
        
        return {
            'category': category,
            'level': level,
            'score': category_scores[best_category] / len(query_words)
        }


# ========================= 코드 생성기 =========================
class CodeGenerator:
    """신규 코드 생성기"""
    
    def __init__(self):
        self.code_patterns = {}
        self.category_codes = defaultdict(list)
        self.last_numbers = defaultdict(int)
        print("코드 생성기 초기화")
    
    def build_patterns(self, df):
        """기존 코드 패턴 학습"""
        print("코드 패턴 학습 중...")
        
        for _, row in df.iterrows():
            code = row.get('P CODE', '')
            if pd.notna(code) and code:
                # 카테고리별 코드 수집
                for level in ['L1 NAME', 'L2 NAME', 'L3 NAME']:
                    if level in row and pd.notna(row[level]):
                        category = str(row[level])
                        self.category_codes[category].append(code)
                
                # 코드 패턴 분석
                self._analyze_code_pattern(code)
        
        print(f"총 {len(self.category_codes)} 개 카테고리의 코드 패턴 학습 완료")
    
    def _analyze_code_pattern(self, code):
        """코드 패턴 분석"""
        # 코드에서 숫자 부분 추출
        match = re.search(r'(\d+)$', code)
        if match:
            prefix = code[:match.start()]
            number = int(match.group(1))
            
            # 프리픽스별 최대 번호 저장
            if prefix not in self.last_numbers or number > self.last_numbers[prefix]:
                self.last_numbers[prefix] = number
    
    def generate_code(self, classification):
        """카테고리 기반 신규 코드 생성"""
        if not classification:
            return self._generate_random_code()
        
        category = classification['category']
        
        # 1. 해당 카테고리의 기존 코드들 확인
        existing_codes = self.category_codes.get(category, [])
        
        if existing_codes:
            # 가장 빈번한 프리픽스 찾기
            prefix_counts = Counter()
            for code in existing_codes:
                match = re.search(r'^([A-Z]+)', code)
                if match:
                    prefix_counts[match.group(1)] += 1
            
            if prefix_counts:
                # 가장 빈번한 프리픽스 사용
                common_prefix = prefix_counts.most_common(1)[0][0]
                
                # 해당 프리픽스의 다음 번호 생성
                pattern = f"^{common_prefix}.*?(\\d+)$"
                max_num = 0
                
                for code in existing_codes:
                    match = re.search(pattern, code)
                    if match:
                        num = int(match.group(1))
                        max_num = max(max_num, num)
                
                # 새 코드 생성
                new_num = max_num + 1
                new_code = f"{common_prefix}{category[:3].upper()}{new_num:03d}"
                
                return new_code
        
        # 2. 카테고리 기반 새 코드 생성
        # 카테고리명의 첫 글자들로 프리픽스 생성
        words = category.split()
        prefix = ''.join([w[0] for w in words[:3] if w])[:3]
        
        # 유니크한 코드 생성
        suffix = f"{np.random.randint(100, 999)}"
        new_code = f"{prefix}{suffix}{category[0]}{np.random.randint(100, 999)}"
        
        return new_code.upper()
    
    def _generate_random_code(self):
        """완전 랜덤 코드 생성"""
        chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        prefix = ''.join(np.random.choice(list(chars), 3))
        suffix = np.random.randint(10000, 99999)
        return f"{prefix}{suffix}"


# ========================= 통합 매칭 시스템 =========================
class ShipSupplyCodeMatcher:
    """통합 매칭 시스템"""
    
    def __init__(self, preprocessor, search_engine, classifier, code_generator):
        self.preprocessor = preprocessor
        self.search_engine = search_engine
        self.classifier = classifier
        self.code_generator = code_generator
        
        # 임계값 설정
        self.retrieval_threshold = 0.3
        self.classification_threshold = 0.5
        
        print("통합 시스템 준비 완료")
    
    def match_code(self, query, verbose=True):
        """메인 매칭 함수"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"검색: {query}")
            print(f"{'='*60}")
        
        # 1. 전처리
        processed_query = self.preprocessor.preprocess_query(query)
        if verbose:
            print(f"전처리: {query} → {processed_query}")
        
        # 2. 벡터 검색
        search_results = self.search_engine.search(processed_query, k=5)
        
        # 디버깅: 검색 결과 상세 정보 출력
        if verbose:
            print(f"\n벡터 검색 결과: {len(search_results)}개 발견")
            if search_results:
                print(f"최고 신뢰도: {search_results[0]['confidence']:.3f}")
                print(f"임계값: {self.retrieval_threshold}")
                # 상위 3개 결과 미리보기
                for i, r in enumerate(search_results[:3]):
                    print(f"  [{i+1}] {r['code']} (신뢰도: {r['confidence']:.3f})")
        
        if search_results and search_results[0]['confidence'] >= self.retrieval_threshold:
            if verbose:
                print(f"\n벡터 검색 성공 (신뢰도: {search_results[0]['confidence']:.3f})")
            
            return {
                'method': 'vector_search',
                'success': True,
                'results': search_results,
                'query': query,
                'processed_query': processed_query
            }
        
        # 3. 분류 시도
        if verbose:
            if search_results:
                print(f"\n벡터 검색 신뢰도 부족 ({search_results[0]['confidence']:.3f} < {self.retrieval_threshold})")
            else:
                print("\n벡터 검색 결과 없음")
            print("분류기로 전환...")
        
        classification = self.classifier.classify(processed_query)
        
        if classification and classification['score'] >= self.classification_threshold:
            if verbose:
                print(f"분류: {classification['category']} (점수: {classification['score']:.3f})")
            
            # 4. 신규 코드 생성
            new_code = self.code_generator.generate_code(classification)
            if verbose:
                print(f"신규 코드 생성: {new_code}")
            
            return {
                'method': 'classification',
                'success': True,
                'category': classification['category'],
                'new_code': new_code,
                'confidence': classification['score'],
                'query': query,
                'processed_query': processed_query
            }
        
        # 5. 모두 실패
        if verbose:
            print("\n매칭 실패")
        
        return {
            'method': 'failed',
            'success': False,
            'message': '더 구체적인 키워드를 입력해주세요.',
            'query': query,
            'processed_query': processed_query
        }
    
    def display_results(self, result):
        """결과를 보기 좋게 출력"""
        if result['success']:
            if result['method'] == 'vector_search':
                print(f"\n검색 결과 (상위 3개):")
                for i, item in enumerate(result['results'][:3], 1):
                    print(f"\n  [{i}] P CODE: {item['code']}")
                    print(f"      경로: {item['path']}")
                    print(f"      신뢰도: {item['confidence']:.3f}")
                    if item['spec']:
                        print(f"      규격: {item['spec']}")
            
            elif result['method'] == 'classification':
                print(f"\n분류 및 생성 결과:")
                print(f"  카테고리: {result['category']}")
                print(f"  신규 코드: {result['new_code']}")
                print(f"  신뢰도: {result['confidence']:.3f}")
        
        else:
            print(f"\n실패: {result['message']}")
    
    def search_code(self, query, k=5):
        """search_code 호환성을 위한 메서드"""
        # match_code를 호출하되 verbose=False로
        result = self.match_code(query, verbose=False)
        
        # 기존 search_code 형식으로 변환
        if result['success'] and result['method'] == 'vector_search':
            return {
                'query': query,
                'processed': result['processed_query'],
                'results': result['results'][:k]
            }
        else:
            return {
                'query': query,
                'processed': result['processed_query'],
                'results': []
            }