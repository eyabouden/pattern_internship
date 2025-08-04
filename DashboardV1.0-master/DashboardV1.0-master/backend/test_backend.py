"""
Test module for the Pattern Analysis Backend
"""
import pandas as pd
import numpy as np
from datetime import datetime
import json
from data_processor import DataProcessor
from pattern_analyzer import PatternAnalyzer
from pattern_miner import GPGrowthPatternMiner, PatternRefiner

def create_sample_data():
    """Create sample data for testing"""
    # Sample CRM data
    crm_data = {
        'sector': ['Technology', 'Healthcare', 'Finance', 'Technology', 'Healthcare'] * 20,
        'team_size': np.random.randint(3, 15, 100),
        'deal_value': np.random.randint(50000, 500000, 100),
        'status': np.random.choice(['won', 'lost', 'pending'], 100),
        'submission_date': pd.date_range('2023-01-01', periods=100, freq='D')
    }
    
    # Sample Financial data
    financial_data = {
        'month': np.random.randint(1, 13, 100),
        'revenue': np.random.randint(100000, 1000000, 100),
        'costs': np.random.randint(50000, 500000, 100),
        'department': np.random.choice(['Sales', 'Marketing', 'Engineering'], 100)
    }
    
    return {
        'crm': pd.DataFrame(crm_data),
        'financial': pd.DataFrame(financial_data)
    }

def test_data_processor():
    """Test the data processor"""
    print("🧪 Testing Data Processor...")
    
    processor = DataProcessor()
    
    # Test data type detection
    assert processor.detect_data_type('crm_data.csv') == 'crm'
    assert processor.detect_data_type('financial_report.xlsx') == 'financial'
    assert processor.detect_data_type('hr_data.json') == 'hr'
    
    print("✅ Data Processor tests passed!")

def test_pattern_miner():
    """Test the pattern miner"""
    print("🧪 Testing Pattern Miner...")
    
    miner = GPGrowthPatternMiner(min_support=0.1, min_confidence=0.6)
    
    # Create sample data
    sample_data = create_sample_data()
    crm_df = sample_data['crm']
    
    # Test preprocessing
    processed_df = miner.preprocess_data(crm_df)
    assert len(processed_df) == len(crm_df)
    
    # Test transaction extraction
    transactions = miner.extract_transactions(processed_df, ['sector', 'team_size'], ['deal_value', 'status'])
    assert len(transactions) > 0
    
    # Test pattern mining
    patterns = miner.mine_patterns(transactions, max_patterns=5)
    print(f"   Found {len(patterns)} patterns")
    
    print("✅ Pattern Miner tests passed!")

def test_pattern_refiner():
    """Test the pattern refiner"""
    print("🧪 Testing Pattern Refiner...")
    
    refiner = PatternRefiner()
    
    # Create a sample raw pattern
    raw_pattern = {
        'antecedent': ['sector_Technology'],
        'consequent': ['status_won'],
        'support': 0.3,
        'confidence': 0.8,
        'lift': 2.5
    }
    
    # Test pattern refinement
    refined_pattern = refiner.refine_pattern(raw_pattern, {
        'data_type': 'crm',
        'record_count': 100,
        'columns': ['sector', 'status', 'deal_value']
    })
    
    assert 'title' in refined_pattern
    assert 'description' in refined_pattern
    assert 'businessValue' in refined_pattern
    assert 'implementation' in refined_pattern
    
    print("✅ Pattern Refiner tests passed!")

def test_pattern_analyzer():
    """Test the pattern analyzer"""
    print("🧪 Testing Pattern Analyzer...")
    
    analyzer = PatternAnalyzer()
    
    # Create sample data sources
    sample_data = create_sample_data()
    
    data_sources = [
        {
            'name': 'crm_data.csv',
            'type': 'crm',
            'data': sample_data['crm'].to_dict('records')
        },
        {
            'name': 'financial_data.csv',
            'type': 'financial',
            'data': sample_data['financial'].to_dict('records')
        }
    ]
    
    # Test pattern analysis
    result = analyzer.analyze_patterns(data_sources, {'maxPatterns': 5})
    
    assert 'patterns' in result
    assert 'analysisMetadata' in result
    assert 'statistics' in result
    
    print(f"   Found {len(result['patterns'])} patterns")
    print(f"   Processed {result['analysisMetadata']['total_records']} records")
    
    print("✅ Pattern Analyzer tests passed!")

def run_all_tests():
    """Run all tests"""
    print("🚀 Running Pattern Analysis Backend Tests...")
    print("=" * 50)
    
    try:
        test_data_processor()
        test_pattern_miner()
        test_pattern_refiner()
        test_pattern_analyzer()
        
        print("=" * 50)
        print("🎉 All tests passed! Backend is ready to use.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == '__main__':
    run_all_tests() 