"""
NLP Log Analyzer
NLTK-based text analysis for log messages:
- Keyword extraction
- Word frequency analysis
- Log summarization
- Sentiment/severity analysis
"""

import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from collections import Counter

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


def _ensure_nltk_data():
    """Download required NLTK data if not present."""
    if not NLTK_AVAILABLE:
        return
    for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger",
                     "averaged_perceptron_tagger_eng"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}" if resource in ["stopwords", "wordnet"] else f"taggers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass


class NLPAnalyzer:
    """NLP-based analysis of log messages."""

    # Log-specific stop words
    LOG_STOP_WORDS = {
        "log", "info", "debug", "warn", "error", "http", "https",
        "get", "post", "put", "delete", "null", "none", "true", "false",
        "com", "org", "net", "www", "localhost", "server", "client",
    }

    # Severity-indicating keywords
    SEVERITY_KEYWORDS = {
        "critical": ["crash", "panic", "fatal", "corruption", "breach", "exhausted", "failure", "killed"],
        "high": ["error", "exception", "failed", "timeout", "refused", "denied", "unauthorized", "unreachable"],
        "medium": ["warning", "slow", "deprecated", "approaching", "nearing", "exceeded", "retry"],
        "low": ["info", "success", "completed", "started", "connected", "loaded", "healthy"],
    }

    def __init__(self, config: dict = None):
        self.config = config or {}
        nlp_config = self.config.get("ml", {}).get("nlp", {})
        self.top_keywords = nlp_config.get("top_keywords", 50)
        self.ngram_range = tuple(nlp_config.get("ngram_range", [1, 2]))

        _ensure_nltk_data()

        if NLTK_AVAILABLE:
            try:
                self.stop_words = set(stopwords.words("english")) | self.LOG_STOP_WORDS
            except Exception:
                self.stop_words = self.LOG_STOP_WORDS
            try:
                self.lemmatizer = WordNetLemmatizer()
            except Exception:
                self.lemmatizer = None
        else:
            self.stop_words = self.LOG_STOP_WORDS
            self.lemmatizer = None

    def analyze(self, messages: pd.Series) -> Dict:
        """
        Perform comprehensive NLP analysis on log messages.

        Args:
            messages: Series of log message strings

        Returns:
            Dictionary with analysis results
        """
        print("\n" + "=" * 60)
        print("NLP LOG ANALYSIS")
        print("=" * 60)

        clean_msgs = messages.fillna("").astype(str)
        clean_msgs = clean_msgs[clean_msgs.str.len() > 0]
        print(f"  Analyzing {len(clean_msgs):,} messages...")

        results = {}

        # 1. Keyword extraction
        print("\n  [1/5] Extracting keywords...")
        results["keywords"] = self._extract_keywords(clean_msgs)

        # 2. Word frequency analysis
        print("  [2/5] Word frequency analysis...")
        results["word_frequencies"] = self._word_frequencies(clean_msgs)

        # 3. N-gram analysis
        print("  [3/5] N-gram analysis...")
        results["ngrams"] = self._ngram_analysis(clean_msgs)

        # 4. Severity analysis
        print("  [4/5] Severity keyword analysis...")
        results["severity_analysis"] = self._severity_analysis(clean_msgs)

        # 5. Log summarization
        print("  [5/5] Generating summary...")
        results["summary"] = self._summarize(clean_msgs, results)

        print(f"\n  [OK] NLP analysis complete")
        return results

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and clean text."""
        # Remove IPs, timestamps, hex values, paths
        text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', text)
        text = re.sub(r'\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}', ' ', text)
        text = re.sub(r'0x[0-9a-fA-F]+', ' ', text)
        text = re.sub(r'[/\\][\w./\\]+', ' ', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = text.lower()

        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text)
            except Exception:
                tokens = text.split()
        else:
            tokens = text.split()

        # Filter
        tokens = [
            t for t in tokens
            if len(t) > 2 and t not in self.stop_words
        ]

        # Lemmatize
        if self.lemmatizer:
            try:
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            except Exception:
                pass

        return tokens

    def _extract_keywords(self, messages: pd.Series) -> List[Dict]:
        """Extract top keywords from messages using frequency and TF approach."""
        all_tokens = []
        for msg in messages:
            all_tokens.extend(self._tokenize(msg))

        # Frequency-based keywords
        counter = Counter(all_tokens)
        total = len(all_tokens)

        keywords = []
        for word, count in counter.most_common(self.top_keywords):
            keywords.append({
                "keyword": word,
                "count": count,
                "frequency": round(count / total * 100, 3)
            })

        print(f"    Found {len(counter)} unique tokens, top {len(keywords)} extracted")
        return keywords

    def _word_frequencies(self, messages: pd.Series) -> Dict:
        """Compute word frequency distribution."""
        all_tokens = []
        for msg in messages:
            all_tokens.extend(self._tokenize(msg))

        counter = Counter(all_tokens)
        total = len(all_tokens)

        return {
            "total_tokens": total,
            "unique_tokens": len(counter),
            "vocabulary_richness": round(len(counter) / max(total, 1), 4),
            "top_20": [{"word": w, "count": c} for w, c in counter.most_common(20)],
        }

    def _ngram_analysis(self, messages: pd.Series) -> Dict:
        """Extract common n-grams from messages."""
        bigrams = Counter()
        trigrams = Counter()

        for msg in messages:
            tokens = self._tokenize(msg)
            for i in range(len(tokens) - 1):
                bigrams[f"{tokens[i]} {tokens[i+1]}"] += 1
            for i in range(len(tokens) - 2):
                trigrams[f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"] += 1

        return {
            "top_bigrams": [{"ngram": ng, "count": c} for ng, c in bigrams.most_common(20)],
            "top_trigrams": [{"ngram": ng, "count": c} for ng, c in trigrams.most_common(15)],
        }

    def _severity_analysis(self, messages: pd.Series) -> Dict:
        """Analyze messages for severity-indicating keywords."""
        severity_counts = {level: 0 for level in self.SEVERITY_KEYWORDS}
        matched_keywords = {level: Counter() for level in self.SEVERITY_KEYWORDS}

        for msg in messages:
            msg_lower = msg.lower()
            for level, keywords in self.SEVERITY_KEYWORDS.items():
                for kw in keywords:
                    if kw in msg_lower:
                        severity_counts[level] += 1
                        matched_keywords[level][kw] += 1

        return {
            "severity_counts": severity_counts,
            "keyword_matches": {
                level: dict(counter.most_common(5))
                for level, counter in matched_keywords.items()
                if counter
            },
        }

    def _summarize(self, messages: pd.Series, analysis_results: Dict) -> Dict:
        """Generate an extractive summary of log messages."""
        total = len(messages)
        severity = analysis_results.get("severity_analysis", {}).get("severity_counts", {})
        keywords = analysis_results.get("keywords", [])

        # Key insights
        insights = []

        critical_count = severity.get("critical", 0)
        high_count = severity.get("high", 0)
        if critical_count > 0:
            insights.append(f"[!] {critical_count} messages contain CRITICAL severity indicators")
        if high_count > 0:
            insights.append(f"[!] {high_count} messages contain HIGH severity indicators")

        top_kw = [k["keyword"] for k in keywords[:5]]
        if top_kw:
            insights.append(f"Most frequent terms: {', '.join(top_kw)}")

        # Unique message templates
        unique_pct = analysis_results.get("word_frequencies", {}).get("vocabulary_richness", 0) * 100
        insights.append(f"Vocabulary richness: {unique_pct:.1f}%")

        return {
            "total_messages": total,
            "insights": insights,
            "generated_at": datetime.now().isoformat(),
        }

    def generate_wordcloud(self, messages: pd.Series, output_path: str = "data/reports/wordcloud.png") -> Optional[str]:
        """Generate a word cloud image from log messages."""
        if not WORDCLOUD_AVAILABLE:
            print("  [!] wordcloud package not available, skipping")
            return None

        all_tokens = []
        for msg in messages:
            all_tokens.extend(self._tokenize(msg))

        if not all_tokens:
            return None

        text = " ".join(all_tokens)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        wc = WordCloud(
            width=1200,
            height=600,
            background_color="#1a1a2e",
            colormap="plasma",
            max_words=100,
            stopwords=self.stop_words,
            prefer_horizontal=0.7,
        )
        wc.generate(text)
        wc.to_file(output_path)
        print(f"  [OK] Word cloud saved to {output_path}")
        return output_path

    def save_results(self, results: Dict, output_dir: str = "data/reports") -> str:
        """Save NLP analysis results to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "nlp_analysis.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  [OK] NLP results saved to {output_path}")
        return output_path
