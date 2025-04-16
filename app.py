from flask import Flask, render_template, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
import re
import numpy as np

app = Flask(__name__)

# Initialize models with publicly available alternatives
def load_models():
    try:
        # Using publicly available models instead
        fluency_analyzer = pipeline("text-classification", model="textattack/bert-base-uncased-CoLA")
        grammar_analyzer = pipeline("text-classification", model="textattack/roberta-base-CoLA")
        sentiment_analyzer = pipeline("sentiment-analysis")
        relevance_analyzer = pipeline("zero-shot-classification", 
                                    model="facebook/bart-large-mnli")
        
        return {
            'fluency': fluency_analyzer,
            'grammar': grammar_analyzer,
            'sentiment': sentiment_analyzer,
            'relevance': relevance_analyzer
        }
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

models = load_models()

class InterviewEvaluator:
    def evaluate_answer(self, question, answer):
        try:
            # Evaluate each parameter
            fluency_score = self._evaluate_fluency(answer)
            relevance_score = self._evaluate_relevance(question, answer)
            confidence_score = self._evaluate_confidence(answer)
            grammar_score = self._evaluate_grammar(answer)
            completeness_score = self._evaluate_completeness(question, answer)
            
            # Generate feedback
            feedback = self._generate_feedback(
                fluency_score, 
                relevance_score, 
                confidence_score, 
                grammar_score, 
                completeness_score
            )
            
            return {
                "fluency": fluency_score,
                "relevance": relevance_score,
                "confidence": confidence_score,
                "grammar": grammar_score,
                "completeness": completeness_score,
                "feedback": feedback
            }
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {
                "error": "An error occurred during evaluation",
                "details": str(e)
            }
    
    def _evaluate_fluency(self, text):
        result = models['fluency'](text)[0]
        score = result['score'] if result['label'] == 'LABEL_1' else 1 - result['score']
        return min(10, round(score * 10))
    
    def _evaluate_grammar(self, text):
        result = models['grammar'](text)[0]
        score = result['score'] if result['label'] == 'LABEL_1' else 1 - result['score']
        return min(10, round(score * 10))
    
    def _evaluate_confidence(self, text):
        result = models['sentiment'](text)[0]
        length_factor = min(1, len(text.split()) / 30)
        if result['label'] == 'POSITIVE':
            return min(10, round((result['score'] * 0.7 + length_factor * 0.3) * 10))
        else:
            return min(10, round(((1 - result['score']) * 0.7 + length_factor * 0.3) * 6))
    
    def _evaluate_relevance(self, question, answer):
        candidate_labels = ["relevant", "somewhat relevant", "irrelevant"]
        result = models['relevance'](question + " [SEP] " + answer, candidate_labels)
        scores = {label:score for label, score in zip(result['labels'], result['scores'])}
        relevance_score = (scores['relevant'] * 1.0 + scores['somewhat relevant'] * 0.6) * 10
        return round(relevance_score)
    
    def _evaluate_completeness(self, question, answer):
        question_type = self._classify_question(question)
        word_count = len(answer.split())
        
        if question_type == "introductory":
            components = 0
            if re.search(r"(my name is|i am)\s+\w+", answer, re.I):
                components += 1
            if re.search(r"(currently|recently|presently)", answer, re.I):
                components += 1
            if re.search(r"(passion|interest|love|enjoy)", answer, re.I):
                components += 1
            if re.search(r"(goal|objective|aspire|want to)", answer, re.I):
                components += 1
            completeness = min(10, components * 2.5 + min(4, word_count / 15))
            
        elif question_type == "experience":
            components = 0
            if re.search(r"(situation|context|when|where)", answer, re.I):
                components += 1
            if re.search(r"(task|responsibility|goal)", answer, re.I):
                components += 1
            if re.search(r"(action|did|implemented|worked)", answer, re.I):
                components += 1
            if re.search(r"(result|outcome|achieved|impact)", answer, re.I):
                components += 1
            completeness = min(10, components * 2 + min(5, word_count / 20))
            
        else:
            completeness = min(10, 6 + min(4, word_count / 20))
            
        return round(completeness)
    
    def _classify_question(self, question):
        question = question.lower()
        if any(q in question for q in ["tell me about yourself", "introduce yourself", "who are you"]):
            return "introductory"
        elif any(q in question for q in ["experience", "worked on", "did you", "tell me about a time"]):
            return "experience"
        elif any(q in question for q in ["why should we hire you", "why this role", "why our company"]):
            return "motivational"
        else:
            return "general"
    
    def _generate_feedback(self, fluency, relevance, confidence, grammar, completeness):
        feedback_parts = []
        
        if fluency >= 8:
            feedback_parts.append("Excellent fluency with smooth delivery.")
        elif fluency >= 5:
            feedback_parts.append("Good fluency overall, some minor improvements possible.")
        else:
            feedback_parts.append("Fluency needs work - practice speaking more naturally.")
        
        if relevance >= 9:
            feedback_parts.append("Highly relevant answer that addresses the question well.")
        elif relevance >= 6:
            feedback_parts.append("Mostly relevant but could be more focused on the question.")
        else:
            feedback_parts.append("Answer strays from the question - stay more on topic.")
        
        if confidence >= 8:
            feedback_parts.append("Confident tone that makes a strong impression.")
        elif confidence >= 5:
            feedback_parts.append("Moderate confidence - try to project more assurance.")
        else:
            feedback_parts.append("Lacks confidence - work on speaking with more conviction.")
        
        if grammar >= 9:
            feedback_parts.append("Perfect grammar usage.")
        elif grammar >= 7:
            feedback_parts.append("Mostly good grammar with minor issues.")
        else:
            feedback_parts.append("Several grammar mistakes - review and practice.")
        
        if completeness >= 9:
            feedback_parts.append("Comprehensive answer covering all aspects.")
        elif completeness >= 6:
            feedback_parts.append("Good answer but could be more complete.")
        else:
            feedback_parts.append("Incomplete answer - expand on your points.")
        
        scores = {
            'fluency': fluency,
            'relevance': relevance,
            'confidence': confidence,
            'grammar': grammar,
            'completeness': completeness
        }
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        
        final_feedback = []
        for aspect, score in sorted_scores[:2]:
            if aspect == 'fluency':
                if score >= 8:
                    final_feedback.append("Fluency is excellent.")
                elif score >= 5:
                    final_feedback.append("Fluency could be slightly improved.")
                else:
                    final_feedback.append("Fluency needs significant improvement.")
            elif aspect == 'relevance':
                if score >= 9:
                    final_feedback.append("Answer is highly relevant.")
                elif score >= 6:
                    final_feedback.append("Answer could be more relevant.")
                else:
                    final_feedback.append("Answer lacks relevance to the question.")
            elif aspect == 'confidence':
                if score >= 8:
                    final_feedback.append("Delivery is confident.")
                elif score >= 5:
                    final_feedback.append("Could show more confidence.")
                else:
                    final_feedback.append("Lacks confidence in delivery.")
            elif aspect == 'grammar':
                if score >= 9:
                    final_feedback.append("Grammar is perfect.")
                elif score >= 7:
                    final_feedback.append("Minor grammar issues.")
                else:
                    final_feedback.append("Grammar needs improvement.")
            elif aspect == 'completeness':
                if score >= 9:
                    final_feedback.append("Answer is very complete.")
                elif score >= 6:
                    final_feedback.append("Answer could be more complete.")
                else:
                    final_feedback.append("Answer is incomplete.")
        
        return " ".join(final_feedback[:3])

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    question = data.get('question', '')
    answer = data.get('answer', '')
    
    if not question or not answer:
        return jsonify({"error": "Both question and answer are required"}), 400
    
    evaluator = InterviewEvaluator()
    evaluation = evaluator.evaluate_answer(question, answer)
    
    return jsonify(evaluation)

if __name__ == '__main__':
    app.run(debug=True)