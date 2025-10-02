import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from . import models

# Hair Health Nutrition Database
HAIR_NUTRIENTS = {
    "protein": {
        "importance": "Hair is 95% protein (keratin)",
        "daily_need_base": 0.8,  # g per kg body weight
        "hair_loss_multiplier": 1.3,
        "sources": ["lean meats", "fish", "eggs", "legumes", "quinoa", "greek yogurt"]
    },
    "iron": {
        "importance": "Most common deficiency in hair loss",
        "daily_need_base": 18,  # mg for women, 8 for men
        "hair_loss_multiplier": 1.5,
        "sources": ["red meat", "spinach", "lentils", "pumpkin seeds", "dark chocolate"]
    },
    "biotin": {
        "importance": "B-vitamin essential for hair growth",
        "daily_need_base": 30,  # mcg
        "hair_loss_multiplier": 100,  # therapeutic dose
        "sources": ["eggs", "nuts", "seeds", "sweet potatoes", "avocados"]
    },
    "zinc": {
        "importance": "Deficiency linked to hair thinning",
        "daily_need_base": 8,  # mg for women, 11 for men
        "hair_loss_multiplier": 1.5,
        "sources": ["oysters", "beef", "pumpkin seeds", "chickpeas", "cashews"]
    },
    "omega3": {
        "importance": "Anti-inflammatory, promotes hair density",
        "daily_need_base": 1.1,  # g for women, 1.6 for men
        "hair_loss_multiplier": 2.0,
        "sources": ["fatty fish", "walnuts", "flaxseeds", "chia seeds", "algae oil"]
    },
    "vitamin_d": {
        "importance": "Essential for hair follicle health",
        "daily_need_base": 600,  # IU
        "hair_loss_multiplier": 2.0,
        "sources": ["fatty fish", "fortified foods", "egg yolks", "mushrooms"]
    },
    "vitamin_c": {
        "importance": "Iron absorption and collagen synthesis",
        "daily_need_base": 75,  # mg for women, 90 for men
        "hair_loss_multiplier": 1.2,
        "sources": ["citrus fruits", "berries", "bell peppers", "broccoli", "kiwi"]
    }
}

# Hair stage to severity mapping
HAIR_STAGE_SEVERITY = {
    "LEVEL_2": 0.2,
    "LEVEL_3": 0.4,
    "LEVEL_4": 0.6,
    "LEVEL_5": 0.8,
    "LEVEL_6": 0.9,
    "LEVEL_7": 1.0
}

class KeratrackDietAI:
    def __init__(self):
        self.nutrition_db = HAIR_NUTRIENTS
        self.stage_severity = HAIR_STAGE_SEVERITY
        
    def analyze_user_hair_progression(self, db: Session, user_id: int) -> Dict:
        """Analyze user's hair loss progression from prediction history"""
        predictions = db.query(models.Prediction).filter(
            models.Prediction.user_id == user_id
        ).order_by(models.Prediction.created_at.desc()).limit(6).all()
        
        if len(predictions) < 2:
            return {
                "progression_rate": 0,
                "current_severity": 0.5,
                "trend": "insufficient_data"
            }
        
        # Convert stages to numeric severity
        severities = [self.stage_severity.get(pred.predicted_stage, 0.5) for pred in predictions]
        dates = [pred.created_at for pred in predictions]
        
        # Calculate progression rate (change per month)
        if len(severities) >= 2:
            days_diff = (dates[0] - dates[-1]).days
            severity_diff = severities[0] - severities[-1]
            progression_rate = (severity_diff / max(days_diff, 1)) * 30  # per month
        else:
            progression_rate = 0
        
        # Determine trend
        if progression_rate > 0.05:
            trend = "worsening"
        elif progression_rate < -0.05:
            trend = "improving"
        else:
            trend = "stable"
            
        return {
            "progression_rate": progression_rate,
            "current_severity": severities[0] if severities else 0.5,
            "trend": trend,
            "confidence": min(len(predictions) / 6.0, 1.0)
        }
    
    def calculate_personalized_nutrients(self, user_data: Dict, hair_analysis: Dict) -> Dict:
        """Calculate personalized nutrient requirements"""
        base_multiplier = 1.0
        age = user_data.get('age', 30)
        gender = user_data.get('gender', 'female')
        weight = user_data.get('weight', 65)
        activity_level = user_data.get('activity_level', 'moderate')
        
        # Adjust based on hair loss severity
        hair_severity = hair_analysis.get('current_severity', 0.5)
        hair_multiplier = 1.0 + hair_severity
        
        # Adjust based on progression rate
        progression_rate = hair_analysis.get('progression_rate', 0)
        if progression_rate > 0:  # Getting worse
            urgency_multiplier = 1.2
        else:
            urgency_multiplier = 1.0
        
        # Calculate personalized nutrients
        personalized_nutrients = {}
        for nutrient, info in self.nutrition_db.items():
            base_need = info['daily_need_base']
            
            # Gender adjustments
            if nutrient == 'iron' and gender == 'male':
                base_need = 8
            elif nutrient == 'omega3' and gender == 'male':
                base_need = 1.6
            elif nutrient == 'vitamin_c' and gender == 'male':
                base_need = 90
                
            # Weight adjustment for protein
            if nutrient == 'protein':
                base_need = base_need * weight
                
            # Apply multipliers
            final_need = base_need * info['hair_loss_multiplier'] * hair_multiplier * urgency_multiplier
            
            personalized_nutrients[nutrient] = {
                'daily_target': round(final_need, 2),
                'unit': self._get_nutrient_unit(nutrient),
                'sources': info['sources'],
                'importance': info['importance']
            }
            
        return personalized_nutrients
    
    def _get_nutrient_unit(self, nutrient: str) -> str:
        """Get the unit for each nutrient"""
        units = {
            'protein': 'g',
            'iron': 'mg',
            'biotin': 'mcg',
            'zinc': 'mg',
            'omega3': 'g',
            'vitamin_d': 'IU',
            'vitamin_c': 'mg'
        }
        return units.get(nutrient, 'units')
    
    def generate_meal_plan(self, nutrients: Dict, dietary_restrictions: List[str]) -> Dict:
        """Generate a weekly meal plan based on nutrient requirements"""
        
        # Sample meal templates (you can expand this with a larger database)
        meal_templates = {
            "high_protein_breakfast": {
                "name": "Hair-Healthy Scrambled Eggs",
                "ingredients": ["3 eggs", "spinach", "avocado", "whole grain toast"],
                "nutrients": {"protein": 25, "iron": 3, "biotin": 25, "omega3": 0.3}
            },
            "iron_rich_lunch": {
                "name": "Lentil and Spinach Salad",
                "ingredients": ["lentils", "spinach", "pumpkin seeds", "bell peppers"],
                "nutrients": {"protein": 18, "iron": 6, "zinc": 3, "vitamin_c": 80}
            },
            "omega3_dinner": {
                "name": "Grilled Salmon with Sweet Potato",
                "ingredients": ["salmon fillet", "sweet potato", "broccoli", "olive oil"],
                "nutrients": {"protein": 35, "omega3": 2.0, "biotin": 5, "vitamin_d": 400}
            }
        }
        
        # Filter based on dietary restrictions
        allowed_meals = self._filter_meals_by_restrictions(meal_templates, dietary_restrictions)
        
        # Generate 7-day plan
        weekly_plan = {}
        for day in range(1, 8):
            daily_plan = {
                "breakfast": self._select_best_meal(allowed_meals, "breakfast", nutrients),
                "lunch": self._select_best_meal(allowed_meals, "lunch", nutrients),
                "dinner": self._select_best_meal(allowed_meals, "dinner", nutrients),
                "snacks": self._recommend_snacks(nutrients, dietary_restrictions)
            }
            weekly_plan[f"day_{day}"] = daily_plan
            
        return weekly_plan
    
    def _filter_meals_by_restrictions(self, meals: Dict, restrictions: List[str]) -> Dict:
        """Filter meals based on dietary restrictions"""
        # Simple filtering logic (expand based on your needs)
        filtered_meals = {}
        for meal_id, meal in meals.items():
            include_meal = True
            
            if "vegetarian" in restrictions:
                if any(meat in str(meal['ingredients']).lower() 
                       for meat in ["salmon", "beef", "chicken", "fish"]):
                    include_meal = False
                    
            if "gluten_free" in restrictions:
                if "toast" in str(meal['ingredients']).lower():
                    include_meal = False
                    
            if include_meal:
                filtered_meals[meal_id] = meal
                
        return filtered_meals if filtered_meals else meals
    
    def _select_best_meal(self, meals: Dict, meal_type: str, target_nutrients: Dict) -> Dict:
        """Select the best meal for nutritional targets"""
        # Simple selection logic - you can make this more sophisticated
        best_meal = None
        best_score = 0
        
        for meal_id, meal in meals.items():
            if meal_type in meal_id:
                score = self._calculate_nutrition_score(meal, target_nutrients)
                if score > best_score:
                    best_score = score
                    best_meal = meal
                    
        return best_meal if best_meal else list(meals.values())[0]
    
    def _calculate_nutrition_score(self, meal: Dict, targets: Dict) -> float:
        """Calculate how well a meal meets nutritional targets"""
        score = 0
        meal_nutrients = meal.get('nutrients', {})
        
        for nutrient, target_info in targets.items():
            target = target_info['daily_target'] / 3  # Assuming 3 main meals
            actual = meal_nutrients.get(nutrient, 0)
            
            # Score based on how close to target (without going too far over)
            if actual >= target * 0.8:  # At least 80% of target
                score += 1
            elif actual >= target * 0.5:  # At least 50% of target
                score += 0.5
                
        return score
    
    def _recommend_snacks(self, nutrients: Dict, restrictions: List[str]) -> List[str]:
        """Recommend healthy snacks"""
        snacks = [
            "Handful of walnuts (omega-3)",
            "Greek yogurt with berries (protein, biotin)",
            "Pumpkin seeds (zinc, iron)",
            "Dark chocolate square (iron)",
            "Apple with almond butter (vitamin C, protein)"
        ]
        
        # Filter based on restrictions
        if "vegetarian" in restrictions:
            # All these snacks are vegetarian-friendly
            pass
            
        return snacks[:3]  # Return top 3 recommendations
    
    def recommend_supplements(self, nutrients: Dict, user_data: Dict) -> List[Dict]:
        """Recommend supplements based on nutrient gaps"""
        supplements = []
        
        # High-priority supplements for hair health
        hair_stage = user_data.get('current_hair_stage', 'LEVEL_4')
        severity = self.stage_severity.get(hair_stage, 0.5)
        
        if severity > 0.6:  # Moderate to severe hair loss
            supplements.extend([
                {
                    "name": "Biotin Complex",
                    "dosage": "5000 mcg daily",
                    "reason": "Therapeutic dose for hair growth support",
                    "timing": "With breakfast"
                },
                {
                    "name": "Iron + Vitamin C",
                    "dosage": "25 mg iron with 100 mg vitamin C",
                    "reason": "Address common iron deficiency, vitamin C enhances absorption",
                    "timing": "Between meals for better absorption"
                },
                {
                    "name": "Omega-3 Fish Oil",
                    "dosage": "1000 mg EPA/DHA daily",
                    "reason": "Anti-inflammatory support for scalp health",
                    "timing": "With largest meal"
                }
            ])
        
        # Add zinc if male or severe hair loss
        if user_data.get('gender') == 'male' or severity > 0.7:
            supplements.append({
                "name": "Zinc Picolinate",
                "dosage": "15 mg daily",
                "reason": "Support hair follicle health and hormone balance",
                "timing": "On empty stomach (2 hours after meals)"
            })
            
        return supplements
    
    def generate_complete_recommendation(self, db: Session, user_id: int, assessment_data: Dict) -> Dict:
        """Generate complete AI-powered diet recommendation"""
        
        # Analyze hair progression
        hair_analysis = self.analyze_user_hair_progression(db, user_id)
        
        # Calculate personalized nutrients
        nutrients = self.calculate_personalized_nutrients(assessment_data, hair_analysis)
        
        # Generate meal plan
        dietary_restrictions = assessment_data.get('dietary_restrictions', [])
        meal_plan = self.generate_meal_plan(nutrients, dietary_restrictions)
        
        # Recommend supplements
        supplements = self.recommend_supplements(nutrients, {
            **assessment_data,
            'current_hair_stage': hair_analysis.get('current_stage', 'LEVEL_4')
        })
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(hair_analysis, assessment_data)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(hair_analysis, nutrients, assessment_data)
        
        return {
            "nutrients": nutrients,
            "meal_plan": meal_plan,
            "supplements": supplements,
            "confidence_score": confidence,
            "reasoning": reasoning,
            "hair_analysis": hair_analysis,
            "next_review_date": (datetime.now() + timedelta(days=30)).isoformat()
        }
    
    def _calculate_confidence_score(self, hair_analysis: Dict, assessment_data: Dict) -> float:
        """Calculate confidence in the recommendation"""
        base_confidence = 0.7
        
        # Increase confidence with more hair prediction data
        data_confidence = hair_analysis.get('confidence', 0.5)
        
        # Increase confidence with more complete assessment
        assessment_completeness = len([v for v in assessment_data.values() if v is not None]) / 10
        
        final_confidence = min(base_confidence + (data_confidence * 0.2) + (assessment_completeness * 0.1), 0.95)
        return round(final_confidence, 2)
    
    def _generate_reasoning(self, hair_analysis: Dict, nutrients: Dict, assessment_data: Dict) -> str:
        """Generate human-readable reasoning for recommendations"""
        reasoning_parts = []
        
        # Hair analysis reasoning
        trend = hair_analysis.get('trend', 'stable')
        severity = hair_analysis.get('current_severity', 0.5)
        
        if trend == "worsening":
            reasoning_parts.append(
                "Your hair loss appears to be progressing, so we've increased your nutrient targets to provide therapeutic support."
            )
        elif trend == "improving":
            reasoning_parts.append(
                "Your hair health seems to be improving! We've maintained supportive nutrient levels to continue this positive trend."
            )
        else:
            reasoning_parts.append(
                "Your hair loss appears stable. This plan focuses on maintaining hair health and preventing further progression."
            )
        
        # Severity-based reasoning
        if severity > 0.7:
            reasoning_parts.append(
                "Given the advanced stage of hair loss, we've prioritized protein, biotin, and iron - the most critical nutrients for hair regrowth."
            )
        
        # Gender-specific reasoning
        gender = assessment_data.get('gender', '')
        if gender == 'female':
            reasoning_parts.append(
                "As a woman, you have higher iron needs, especially important since iron deficiency is the leading nutritional cause of hair loss."
            )
        
        return " ".join(reasoning_parts)