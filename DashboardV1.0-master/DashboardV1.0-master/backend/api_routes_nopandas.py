"""
API Routes Module for Pattern Analysis Backend - Pandas Free Version
"""
from flask import request, jsonify
from datetime import datetime
import logging
import os
import sys

# Add the graphRAG directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "graphRAG"))
from graphrag_from_json import GraphRAGFromJSON

# Import MURAG for competitor intelligence
from MURAG import PDFCopilot

from config import UPLOAD_FOLDER

logger = logging.getLogger(__name__)

class APIRoutes:
    """Handles all API routes for the pattern analysis backend"""
    
    def __init__(self, app):
        self.app = app
        
        # Initialize GraphRAG using the new class
        google_api_key = os.getenv('GOOGLE_API_KEY', 'AIzaSyASgbjsK530ZrtX7wqVTPY1Lblcg2YBzQ8')
        json_path = os.path.join(os.path.dirname(__file__), "..", "graphRAG", "case_studies_graph.json")
        self.graph_rag_analyzer = GraphRAGFromJSON(json_path, google_api_key)
        
        # Initialize MURAG for competitor intelligence
        together_api_key = os.getenv('TOGETHER_API_KEY', '')
        if together_api_key:
            self.murag_copilot = PDFCopilot(together_api_key)
        else:
            self.murag_copilot = None
            logger.warning("TOGETHER_API_KEY not set - competitor intelligence disabled")
        
        # Register routes
        self.register_routes()
    
    def register_routes(self):
        """Register all API routes"""
        self.app.add_url_rule('/api/health', 'health_check', self.health_check, methods=['GET'])
        self.app.add_url_rule('/api/analyze-graphrag', 'analyze_graphrag', self.analyze_graphrag, methods=['POST'])
        self.app.add_url_rule('/api/competitor-chat', 'competitor_chat', self.competitor_chat, methods=['POST'])
    
    def health_check(self):
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {
                'graph_rag_analyzer': 'available',
                'competitor_intelligence': 'available' if self.murag_copilot else 'disabled'
            }
        })
    
    def upload_data(self):
        """Handle data file upload"""
        try:
            # Validate request
            is_valid, message = self.data_processor.validate_upload_request(request)
            if not is_valid:
                return jsonify({'success': False, 'error': message}), 400
            
            file = request.files['file']
            
            # Save file
            filepath = self.data_processor.save_uploaded_file(file)
            
            # Process the data
            result = self.data_processor.process_data_for_analysis(filepath)
            
            if result['success']:
                return jsonify({
                    'success': True,
                    'message': 'File uploaded and processed successfully',
                    'data': result
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result.get('error', 'Processing failed')
                }), 500
                
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def analyze_patterns(self):
        """Handle pattern analysis requests"""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'}), 400
            
            # Extract data sources and configuration
            data_sources = data.get('dataSources', [])
            config = data.get('config', {})
            
            if not data_sources:
                return jsonify({'success': False, 'error': 'No data sources provided'}), 400
            
            # Run pattern analysis
            result = self.pattern_analyzer.analyze_patterns(data_sources, config)
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def analyze_graphrag(self):
        """Handle GraphRAG pattern analysis requests using notebook approach"""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'}), 400
            
            # Extract query from request - check both root level and config
            query = data.get('query', '')
            config = data.get('config', {})
            
            # If no query at root level, check inside config
            if not query:
                query = config.get('query', '')
            
            if not query:
                return jsonify({'success': False, 'error': 'No query provided'}), 400
            
            logger.info(f"🚀 Starting GraphRAG analysis with query: {query}")
            
            # Run analysis using GraphRAG from JSON
            result = self.graph_rag_analyzer.answer_question(
                query=query,
                hops=config.get('hops', 1),
                k=config.get('k', 12),
                max_nodes=config.get('max_nodes', 1500)
            )
            
            # Format response for frontend compatibility
            response = {
                'success': True,
                'patterns': result['answer'],
                'analysisMetadata': {
                    'retrievedDocs': result['metadata']['retrieved_docs'],
                    'seedCount': result['metadata']['seed_count'],
                    'subgraphNodes': result['metadata']['subgraph_nodes'],
                    'subgraphEdges': result['metadata']['subgraph_edges'],
                    'query': query,
                    'approach': 'graphrag_from_json'
                },
                'subgraph': {
                    'nodes': [
                        {
                            'id': self.graph_rag_analyzer._n_id(n),
                            'type': ', '.join(self.graph_rag_analyzer._labels(n)),
                            'title': self.graph_rag_analyzer._title(n),
                            'data': self.graph_rag_analyzer._n_data(n)
                        } for n in result['sub_nodes'][:50]  # Limit for response size
                    ],
                    'edges': [
                        {
                            'source': self.graph_rag_analyzer._src(e),
                            'target': self.graph_rag_analyzer._tgt(e),
                            'type': self.graph_rag_analyzer._e_data(e).get('type', '')
                        } for e in result['sub_edges'][:100]  # Limit for response size
                    ]
                }
            }
            
            logger.info(f"✓ Notebook GraphRAG analysis completed successfully")
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"❌ Notebook GraphRAG analysis error: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e),
                'patterns': '',
                'analysisMetadata': {
                    'error': str(e),
                    'approach': 'notebook_graphrag'
                }
            }), 500
    
    def competitor_chat(self):
        """Handle competitor intelligence chat requests using MURAG"""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'}), 400
            
            # Extract query from request
            query = data.get('query', '')
            
            if not query:
                return jsonify({'success': False, 'error': 'No query provided'}), 400
                
            if not self.murag_copilot:
                return jsonify({
                    'success': False, 
                    'error': 'Competitor intelligence service not available. Please set TOGETHER_API_KEY.'
                }), 503
            
            logger.info(f"🔍 Starting competitor intelligence analysis with query: {query}")
            
            # Get response from MURAG
            response_text = self.murag_copilot.ask(query)
            
            # Format response for frontend
            response = {
                'success': True,
                'answer': response_text,
                'timestamp': datetime.now().isoformat(),
                'query': query
            }
            
            logger.info(f"✅ Competitor intelligence analysis completed")
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"❌ Competitor intelligence error: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e),
                'answer': f"Sorry, I encountered an error while analyzing competitor information: {str(e)}"
            }), 500
