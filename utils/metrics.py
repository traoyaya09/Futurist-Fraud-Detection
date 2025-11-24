"""
Metrics and Performance Monitoring Utilities
Handles recommendation quality metrics, A/B testing, and performance tracking
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger("RecommendationService.Metrics")


class RecommendationMetrics:
    """Track and compute recommendation system metrics"""
    
    def __init__(self):
        self.metrics_history = []
        self.session_metrics = defaultdict(list)
    
    def compute_precision_at_k(
        self,
        recommended: List[str],
        relevant: List[str],
        k: int = 10
    ) -> float:
        """
        Compute Precision@K
        
        Args:
            recommended: List of recommended product IDs
            relevant: List of relevant (clicked/purchased) product IDs
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K score
        """
        if not recommended or not relevant:
            return 0.0
        
        top_k = recommended[:k]
        relevant_set = set(relevant)
        
        hits = sum(1 for item in top_k if item in relevant_set)
        precision = hits / min(k, len(top_k))
        
        return precision
    
    def compute_recall_at_k(
        self,
        recommended: List[str],
        relevant: List[str],
        k: int = 10
    ) -> float:
        """
        Compute Recall@K
        
        Args:
            recommended: List of recommended product IDs
            relevant: List of relevant product IDs
            k: Number of top recommendations to consider
            
        Returns:
            Recall@K score
        """
        if not recommended or not relevant:
            return 0.0
        
        top_k = recommended[:k]
        relevant_set = set(relevant)
        
        hits = sum(1 for item in top_k if item in relevant_set)
        recall = hits / len(relevant_set) if relevant_set else 0.0
        
        return recall
    
    def compute_ndcg_at_k(
        self,
        recommended: List[str],
        relevant: List[str],
        relevance_scores: Optional[Dict[str, float]] = None,
        k: int = 10
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain@K
        
        Args:
            recommended: List of recommended product IDs
            relevant: List of relevant product IDs
            relevance_scores: Optional relevance scores for each item
            k: Number of top recommendations to consider
            
        Returns:
            NDCG@K score
        """
        if not recommended or not relevant:
            return 0.0
        
        top_k = recommended[:k]
        relevant_set = set(relevant)
        
        # Compute DCG
        dcg = 0.0
        for i, item in enumerate(top_k):
            if item in relevant_set:
                rel = relevance_scores.get(item, 1.0) if relevance_scores else 1.0
                dcg += rel / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # Compute IDCG (ideal DCG)
        ideal_relevance = sorted(
            [relevance_scores.get(item, 1.0) if relevance_scores else 1.0 for item in relevant_set],
            reverse=True
        )[:k]
        
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        # Compute NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return ndcg
    
    def compute_map_at_k(
        self,
        recommended: List[str],
        relevant: List[str],
        k: int = 10
    ) -> float:
        """
        Compute Mean Average Precision@K
        
        Args:
            recommended: List of recommended product IDs
            relevant: List of relevant product IDs
            k: Number of top recommendations to consider
            
        Returns:
            MAP@K score
        """
        if not recommended or not relevant:
            return 0.0
        
        top_k = recommended[:k]
        relevant_set = set(relevant)
        
        precisions = []
        hits = 0
        
        for i, item in enumerate(top_k):
            if item in relevant_set:
                hits += 1
                precisions.append(hits / (i + 1))
        
        if not precisions:
            return 0.0
        
        return sum(precisions) / len(relevant_set)
    
    def compute_diversity(
        self,
        recommended_products: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute diversity metrics for recommendations
        
        Args:
            recommended_products: List of recommended product documents
            
        Returns:
            Dictionary with diversity metrics
        """
        if not recommended_products:
            return {"category_diversity": 0.0, "brand_diversity": 0.0, "price_diversity": 0.0}
        
        categories = [p.get("category", "") for p in recommended_products]
        brands = [p.get("brand", "") for p in recommended_products]
        prices = [p.get("price", 0) for p in recommended_products if p.get("price", 0) > 0]
        
        # Category diversity (ratio of unique categories)
        category_diversity = len(set(categories)) / len(categories) if categories else 0.0
        
        # Brand diversity
        brand_diversity = len(set(brands)) / len(brands) if brands else 0.0
        
        # Price diversity (coefficient of variation)
        if prices and len(prices) > 1:
            price_std = np.std(prices)
            price_mean = np.mean(prices)
            price_diversity = price_std / price_mean if price_mean > 0 else 0.0
        else:
            price_diversity = 0.0
        
        return {
            "category_diversity": category_diversity,
            "brand_diversity": brand_diversity,
            "price_diversity": price_diversity,
            "overall_diversity": (category_diversity + brand_diversity) / 2
        }
    
    def compute_novelty(
        self,
        recommended: List[str],
        popular_items: List[str],
        top_k: int = 100
    ) -> float:
        """
        Compute novelty (how unexpected the recommendations are)
        
        Args:
            recommended: List of recommended product IDs
            popular_items: List of popular product IDs
            top_k: Number of top popular items to consider
            
        Returns:
            Novelty score (0.0 to 1.0)
        """
        if not recommended:
            return 0.0
        
        top_popular = set(popular_items[:top_k])
        non_popular = sum(1 for item in recommended if item not in top_popular)
        
        novelty = non_popular / len(recommended)
        return novelty
    
    def compute_coverage(
        self,
        recommended_items: List[str],
        all_items: List[str]
    ) -> float:
        """
        Compute catalog coverage (percentage of items recommended)
        
        Args:
            recommended_items: All items that have been recommended
            all_items: All available items in catalog
            
        Returns:
            Coverage ratio (0.0 to 1.0)
        """
        if not all_items:
            return 0.0
        
        recommended_set = set(recommended_items)
        all_items_set = set(all_items)
        
        coverage = len(recommended_set & all_items_set) / len(all_items_set)
        return coverage
    
    def compute_serendipity(
        self,
        recommended: List[str],
        user_profile: List[str],
        relevant: List[str]
    ) -> float:
        """
        Compute serendipity (unexpected yet relevant recommendations)
        
        Args:
            recommended: List of recommended product IDs
            user_profile: User's past interactions
            relevant: Relevant items (actually interacted with)
            
        Returns:
            Serendipity score
        """
        if not recommended or not relevant:
            return 0.0
        
        user_profile_set = set(user_profile)
        relevant_set = set(relevant)
        
        # Items that are relevant but not similar to user profile
        unexpected_relevant = [
            item for item in recommended
            if item in relevant_set and item not in user_profile_set
        ]
        
        serendipity = len(unexpected_relevant) / len(recommended)
        return serendipity


class PerformanceTracker:
    """Track system performance metrics"""
    
    def __init__(self):
        self.request_times = []
        self.error_count = 0
        self.success_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def record_request(self, duration: float, success: bool = True):
        """Record a request"""
        self.request_times.append(duration)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def record_cache_hit(self):
        """Record a cache hit"""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss"""
        self.cache_misses += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.request_times:
            return {
                "total_requests": 0,
                "success_rate": 0.0,
                "avg_response_time": 0.0,
                "p50_response_time": 0.0,
                "p95_response_time": 0.0,
                "p99_response_time": 0.0,
                "cache_hit_rate": 0.0
            }
        
        total_requests = self.success_count + self.error_count
        success_rate = self.success_count / total_requests if total_requests > 0 else 0.0
        
        sorted_times = sorted(self.request_times)
        n = len(sorted_times)
        
        return {
            "total_requests": total_requests,
            "success_rate": success_rate,
            "error_count": self.error_count,
            "avg_response_time": np.mean(sorted_times),
            "p50_response_time": sorted_times[int(n * 0.5)],
            "p95_response_time": sorted_times[int(n * 0.95)],
            "p99_response_time": sorted_times[int(n * 0.99)],
            "min_response_time": min(sorted_times),
            "max_response_time": max(sorted_times),
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) 
                             if (self.cache_hits + self.cache_misses) > 0 else 0.0
        }
    
    def reset(self):
        """Reset all metrics"""
        self.request_times = []
        self.error_count = 0
        self.success_count = 0
        self.cache_hits = 0
        self.cache_misses = 0


class ABTestTracker:
    """Track A/B test results"""
    
    def __init__(self):
        self.variants = defaultdict(lambda: {"impressions": 0, "clicks": 0, "conversions": 0})
    
    def record_impression(self, variant: str):
        """Record an impression"""
        self.variants[variant]["impressions"] += 1
    
    def record_click(self, variant: str):
        """Record a click"""
        self.variants[variant]["clicks"] += 1
    
    def record_conversion(self, variant: str):
        """Record a conversion"""
        self.variants[variant]["conversions"] += 1
    
    def get_results(self) -> Dict[str, Dict[str, float]]:
        """Get A/B test results"""
        results = {}
        
        for variant, data in self.variants.items():
            impressions = data["impressions"]
            clicks = data["clicks"]
            conversions = data["conversions"]
            
            ctr = clicks / impressions if impressions > 0 else 0.0
            cvr = conversions / clicks if clicks > 0 else 0.0
            
            results[variant] = {
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "ctr": ctr,
                "cvr": cvr,
                "overall_conversion_rate": conversions / impressions if impressions > 0 else 0.0
            }
        
        return results
    
    def compute_statistical_significance(
        self,
        variant_a: str,
        variant_b: str,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Compute statistical significance between two variants
        
        Args:
            variant_a: First variant name
            variant_b: Second variant name
            confidence_level: Confidence level (default 0.95)
            
        Returns:
            Dictionary with significance test results
        """
        data_a = self.variants[variant_a]
        data_b = self.variants[variant_b]
        
        n_a = data_a["impressions"]
        n_b = data_b["impressions"]
        
        conv_a = data_a["conversions"]
        conv_b = data_b["conversions"]
        
        p_a = conv_a / n_a if n_a > 0 else 0
        p_b = conv_b / n_b if n_b > 0 else 0
        
        # Pooled proportion
        p_pool = (conv_a + conv_b) / (n_a + n_b) if (n_a + n_b) > 0 else 0
        
        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b)) if n_a > 0 and n_b > 0 else 0
        
        # Z-score
        z_score = (p_b - p_a) / se if se > 0 else 0
        
        # P-value (two-tailed)
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        is_significant = p_value < (1 - confidence_level)
        
        return {
            "variant_a": variant_a,
            "variant_b": variant_b,
            "conversion_rate_a": p_a,
            "conversion_rate_b": p_b,
            "relative_improvement": ((p_b - p_a) / p_a * 100) if p_a > 0 else 0,
            "z_score": z_score,
            "p_value": p_value,
            "is_significant": is_significant,
            "confidence_level": confidence_level
        }


def compute_click_through_rate(impressions: int, clicks: int) -> float:
    """Compute CTR"""
    return clicks / impressions if impressions > 0 else 0.0


def compute_conversion_rate(clicks: int, conversions: int) -> float:
    """Compute conversion rate"""
    return conversions / clicks if clicks > 0 else 0.0


def compute_average_order_value(total_revenue: float, num_orders: int) -> float:
    """Compute AOV"""
    return total_revenue / num_orders if num_orders > 0 else 0.0


def compute_revenue_per_impression(total_revenue: float, impressions: int) -> float:
    """Compute RPI"""
    return total_revenue / impressions if impressions > 0 else 0.0
