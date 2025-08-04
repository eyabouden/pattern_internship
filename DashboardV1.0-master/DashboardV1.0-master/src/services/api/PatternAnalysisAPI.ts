import { DataSource } from '../dataProcessing/DataMerger';

export interface PatternAnalysisConfig {
  maxPatterns: number;
  minConfidence: number;
  minSupport: number;
  complexityLevel: 'medium' | 'high' | 'expert';
  focusAreas: string[];
}

export interface PatternAnalysisResult {
  patterns: ScoredPattern[];
  analysisMetadata: {
    totalRecords: number;
    dataSources: number;
    patternsDetected: number;
    confidence: number;
    timestamp: string;
    processingTime: number;
  };
  statistics: {
    scoring: ScoringStatistics;
    validation: {
      valid: number;
      invalid: number;
    };
  };
}

export interface ScoredPattern {
  id: string;
  title: string;
  description: string;
  impact: number;
  confidence: number;
  complexity: 'Low' | 'Medium' | 'High' | 'Very High';
  variables: string[];
  businessValue: string[];
  implementation: string[];
  metrics: string[];
  scores: PatternScore;
}

export interface PatternScore {
  patternId: string;
  support: number;
  confidence: number;
  lift: number;
  conviction: number;
  impact: number;
  complexity: number;
  overallScore: number;
}

export interface ScoringStatistics {
  totalPatterns: number;
  averageScores: {
    support: number;
    confidence: number;
    lift: number;
    conviction: number;
    impact: number;
    overall: number;
  };
  scoreDistribution: {
    excellent: number;
    good: number;
    fair: number;
    poor: number;
  };
  complexityDistribution: {
    low: number;
    medium: number;
    high: number;
    veryHigh: number;
  };
}

export class PatternAnalysisAPI {
  private baseURL: string;

  constructor() {
    // URL du backend Python Flask
    this.baseURL = 'http://localhost:5000/api';
  }

  /**
   * Upload des fichiers vers le backend Python
   */
  public async uploadFiles(files: File[]): Promise<any> {
    const formData = new FormData();
    
    files.forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await fetch(`${this.baseURL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Upload error:', error);
      throw error;
    }
  }

  /**
   * Analyse des patterns via le backend Python
   */
  public async analyzePatterns(dataSources: DataSource[], config: PatternAnalysisConfig): Promise<PatternAnalysisResult> {
    try {
      const response = await fetch(`${this.baseURL}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          dataSources: dataSources.map(source => ({
            name: source.name,
            type: source.type,
            data: source.data
          })),
          config: config
        }),
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Analysis error:', error);
      throw error;
    }
  }

  /**
   * Vérification de la santé du backend
   */
  public async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseURL}/health`);
      return response.ok;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }

  /**
   * Export des patterns en JSON
   */
  public exportPatternsToJSON(patterns: ScoredPattern[]): string {
    const exportData = {
      patterns: patterns,
      exportMetadata: {
        exportedAt: new Date().toISOString(),
        totalPatterns: patterns.length,
        version: '1.0.0'
      }
    };
    
    return JSON.stringify(exportData, null, 2);
  }

  /**
   * Import des patterns depuis JSON
   */
  public importPatternsFromJSON(jsonData: string): ScoredPattern[] {
    try {
      const data = JSON.parse(jsonData);
      return data.patterns || [];
    } catch (error) {
      console.error('Import error:', error);
      return [];
    }
  }

  /**
   * Génération d'un rapport d'analyse
   */
  public generateAnalysisReport(result: PatternAnalysisResult): {
    summary: string;
    keyInsights: string[];
    recommendations: string[];
    technicalDetails: any;
  } {
    const summary = `Analysis completed with ${result.patterns.length} patterns detected from ${result.analysisMetadata.dataSources} data sources. Overall confidence: ${result.analysisMetadata.confidence.toFixed(1)}%.`;

    const keyInsights = result.patterns.slice(0, 3).map(pattern => 
      `${pattern.title}: ${pattern.description}`
    );

    const recommendations = [
      'Focus on high-impact patterns first',
      'Implement patterns with highest confidence scores',
      'Monitor pattern performance over time',
      'Consider seasonal and temporal patterns in planning'
    ];

    return {
      summary,
      keyInsights,
      recommendations,
      technicalDetails: {
        analysisMetadata: result.analysisMetadata,
        statistics: result.statistics,
        patternBreakdown: {
          byComplexity: result.statistics.scoring.complexityDistribution,
          byQuality: result.statistics.scoring.scoreDistribution
        }
      }
    };
  }
}

export default PatternAnalysisAPI; 