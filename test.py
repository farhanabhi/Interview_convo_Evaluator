def evaluate_performance():
    """Returns fake perfect metrics for demonstration purposes"""
    return {
        'accuracy': 0.9664,
        'precision': {
            'fluency': 0.95,
            'relevance': 0.97,
            'confidence': 0.94,
            'grammar': 0.98,
            'completeness': 0.96
        },
        'f1_scores': {
            'fluency': 0.95,
            'relevance': 0.96,
            'confidence': 0.94,
            'grammar': 0.97,
            'completeness': 0.95
        },
        'message': "NOTE: These are demonstration values only - replace with actual test results"
    }

# Usage
if __name__ == "__main__":
    metrics = evaluate_performance()
    
    print(f"System Accuracy: {metrics['accuracy']*100:.1f}%")
    print("\nPrecision Scores:")
    for metric, score in metrics['precision'].items():
        print(f"{metric.capitalize()}: {score:.2f}")
    
    print(f"\n{metrics['message']}")