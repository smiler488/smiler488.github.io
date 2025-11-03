# AI Solver Tutorial

## Overview

The AI Solver is an advanced multimodal problem-solving tool powered by Hunyuan AI that provides comprehensive solutions across 18 professional domains. This application supports camera capture, screen capture with advanced selection tools, and direct text queries with specialized modes for academic, technical, and creative applications.

## Key Features

- **Multi-modal Input Support**: Camera capture, advanced screen capture with 8-directional selection, and text-based queries
- **18 Professional Presets**: Intelligent analysis, math, physics, chemistry, plant identification, lab safety, translation, English learning, code analysis, text extraction, biology, history, medical, engineering, finance, art, legal, and education
- **Advanced Selection Tools**: Free-form selection with corner and edge resizing for precise area capture
- **Flexible API Configuration**: Default API support with custom endpoint options
- **Step-by-Step Solutions**: Detailed explanations with comprehensive reasoning
- **Hunyuan AI Integration**: Advanced AI capabilities for complex problem solving
- **Real-time Analysis**: Instant processing and response generation
- **Cross-disciplinary Applications**: Comprehensive support for science, technology, engineering, mathematics, arts, and humanities

## Quick Start

### 1. Access the Application

Visit in your browser: `/app/solver`

### 2. System Requirements

- **Modern Web Browser**: Chrome, Firefox, Safari, or Edge with camera support
- **Camera Access**: For image capture functionality (optional)
- **Internet Connection**: Required for AI processing
- **Image Quality**: Clear, well-lit images for optimal analysis

## Detailed Usage Steps

### Step 1: Input Method Selection

1. **Camera Capture**
   - Select "Camera Capture" mode
   - Grant camera permissions when prompted
   - Capture clear, focused images of problems or objects
   - Ensure adequate lighting and minimal shadows
   - Click "Capture and Solve" to process the image

2. **Screen Capture with Advanced Selection**
   - Select "Screen Capture" mode
   - Click "Capture Screen" to start screen sharing
   - Grant screen sharing permissions
   - **Advanced Selection Features**:
     - **Free Movement**: Drag the selection box to move it anywhere
     - **8-Directional Resizing**: Use corner handles (↖ ↗ ↙ ↘) for diagonal resizing
     - **Edge Resizing**: Use side handles (↑ ↓ ← →) for proportional resizing
     - **Minimum Size**: Selection maintains 50px minimum dimensions
     - **Boundary Detection**: Automatic boundary constraints
   - Click "Analyze Selected Area" to process the selected region

3. **Text Input**
   - Select "Text Question" mode
   - Enter questions directly in the text area
   - **Quick Preset Switching**: Type `/preset name` to switch modes (e.g., `/preset Math Problem Solver`)
   - Use detailed descriptions for complex queries
   - Include relevant context for accurate analysis
   - Click "Ask and Get Answer" to submit

### Step 2: Problem Type Selection (18 Professional Presets)

1. **Intelligent Analysis Assistant** (Default)
   - General multimodal analysis for any problem type
   - Step-by-step reasoning with comprehensive explanations
   - Suitable for undefined or mixed problem scenarios

2. **Math Problem Solver**
   - Mathematical equations, algebra, geometry, and calculus
   - Rigorous derivations with elegant solution methods
   - Multiple approach consideration with pros/cons analysis

3. **Physics Problem Solver**
   - Mechanics, electromagnetism, thermodynamics, and quantum physics
   - Fundamental law applications with correct unit handling
   - Experimental uncertainty estimation and boundary analysis

4. **Chemistry Problem Solver**
   - Chemical reactions, stoichiometry, and molecular structures
   - Balanced equations and reaction mechanisms
   - Laboratory procedure analysis and safety considerations

5. **Plant Identification Analysis**
   - Species identification from botanical images
   - Scientific and common names with distinguishing features
   - Growth conditions, native ranges, and care recommendations

6. **Laboratory Safety Assessment**
   - Comprehensive safety evaluation for laboratory environments
   - Chemical, biological, physical, and electrical hazard identification
   - Priority-based corrective measures and PPE recommendations

7. **Academic Translation**
   - Professional translation of academic literature
   - Technical terminology preservation and glossary creation
   - Scholarly style maintenance with structural integrity

8. **English Learning Assistant**
   - Grammar analysis, vocabulary explanations, and exercise solutions
   - Natural example sentences and usage guidance
   - Practice item generation for skill reinforcement

9. **Code Review and Optimization**
   - Code functionality explanation and bug identification
   - Performance optimization and best practice recommendations
   - Complexity analysis and testing suggestions

10. **Text Extraction**
    - Accurate OCR-based text recognition from images
    - Paragraph, table, and mathematical notation preservation
    - Uncertainty marking and reshoot recommendations

11. **Biology Analysis**
    - Biological structure and process identification
    - Cellular, molecular, and ecological relationship analysis
    - Observation and experiment suggestions

12. **Historical Document Interpretation**
    - Artifact and document historical context analysis
    - Era identification and cultural significance interpretation
    - Provenance and dating evidence requirements

13. **Medical Image Analysis** (Informational Only)
    - Anatomical structure identification in medical images
    - Observable findings and general clinical significance
    - **Disclaimer**: Not for diagnostic purposes - consult healthcare professionals

14. **Engineering Drawing Interpretation**
    - Mechanical, civil, electrical, and process engineering drawings
    - Symbol decoding, dimension analysis, and tolerance interpretation
    - Design intent and manufacturing considerations

15. **Financial Statement Analysis** (Informational Only)
    - Performance metrics, trends, and risk assessment
    - Growth, margin, and liquidity analysis
    - **Disclaimer**: For educational purposes only

16. **Artwork Appreciation**
    - Artistic style, technique, and composition analysis
    - Color, texture, medium, and influence interpretation
    - Aesthetic value and contextual meaning discussion

17. **Legal Document Interpretation** (Informational Only)
    - Rights, obligations, and risk point explanation
    - Key clause analysis and ambiguity identification
    - **Disclaimer**: Not legal advice - consult qualified attorneys

18. **Educational Material Analysis**
    - Learning objective and pedagogical approach evaluation
    - Target audience assessment and improvement suggestions
    - Teaching activity and assessment design recommendations

19. **Custom Mode**
    - Flexible handling of special or mixed requests
    - General reasoning rules for unique problem types
    - Alternative solution path consideration
    - Enables manual editing of the question/prompt field (other presets apply their instructions automatically)

### Step 3: API Configuration and Analysis

1. **API Settings**
   - **Default API**: Toggle "Use Default API" to call the same-origin `/api/solve` endpoint (on localhost this targets `http://localhost:3001/api/solve`)
   - **Automatic Fallback**: If the default endpoint responds with 403/404/405, the app shows a notice and returns a mock answer so you can keep testing
   - **Custom API**: Enter a full URL (including protocol) when you deploy your own proxy or serverless function
   - **Security**: No API keys are stored in the browser; credentials must live on the backend
   - **Model Selection**: Choose the appropriate Hunyuan AI model for your use case

2. **Preset Management**
   - **Quick Selection**: Dropdown menu with 18 professional presets
   - **Description Display**: Each preset shows a brief functional description; default prompts are applied automatically
   - **Custom Mode**: Choose this mode if you need to edit the question/prompt text manually
   - **Text Command**: Type `/preset name` in text mode for quick switching

3. **AI Processing**
   - Upload image or text to Hunyuan AI
   - Automatic problem type recognition based on selected preset
   - Context-aware analysis and solution generation
   - Real-time token usage tracking and performance metrics

4. **Step-by-Step Solutions**
   - Comprehensive reasoning process with structured output
   - Detailed explanation of each step with alternative approaches
   - Final answer verification and uncertainty identification
   - Next action suggestions for further exploration

### Step 4: Result Utilization

1. **Solution Review**
   - Carefully review generated solutions
   - Verify accuracy and completeness
   - Cross-reference with known methods

2. **Learning Application**
   - Use solutions for educational purposes
   - Understand underlying principles and concepts
   - Apply similar approaches to related problems

3. **Documentation**
   - Save important solutions for future reference
   - Create personal knowledge base
   - Share insights with colleagues or students

## Technical Specifications

### Input Requirements

#### Image Quality Standards
- **Resolution**: Minimum 640×480 pixels, recommended 1920×1080 or higher
- **Format**: JPEG, PNG, WebP supported
- **Lighting**: Even illumination with minimal shadows
- **Focus**: Sharp, clear images for accurate analysis
- **Contrast**: High contrast between text/objects and background

#### Text Input Guidelines
- **Clarity**: Clear, unambiguous problem statements
- **Context**: Sufficient background information
- **Specificity**: Well-defined problem parameters
- **Completeness**: Include all necessary details

### AI Processing Capabilities

#### Problem Recognition
- **18-Domain Expertise**: Science, technology, engineering, mathematics, arts, humanities, law, finance, medicine
- **Context Understanding**: Semantic analysis with preset-specific optimization
- **Pattern Recognition**: Intelligent identification across 18 professional domains
- **Multimodal Integration**: Seamless image, text, and screen capture processing

#### Solution Generation
- **Structured Reasoning**: Logical progression with preset-specific frameworks
- **Multiple Perspectives**: Alternative approaches with pros/cons analysis
- **Error Checking**: Comprehensive validation with uncertainty identification
- **Explanation Quality**: Clear, comprehensive explanations with next action suggestions

#### Advanced Features
- **8-Directional Selection**: Precise area selection with corner and edge resizing
- **Real-time Token Tracking**: Usage metrics for performance optimization
- **Flexible API Configuration**: Default and custom endpoint support
- **Quick Preset Switching**: Text commands and dropdown selection

### Specialized Application Areas

#### Plant Science and Agriculture
- **Species Identification**: Accurate plant classification
- **Growth Analysis**: Developmental stage assessment
- **Disease Diagnosis**: Symptom recognition and treatment suggestions
- **Crop Management**: Agricultural best practices

#### Educational Applications
- **Homework Assistance**: Problem-solving support
- **Concept Explanation**: Fundamental principle clarification
- **Exam Preparation**: Practice problem analysis
- **Learning Reinforcement**: Knowledge consolidation

#### Professional Applications
- **Research Support**: Literature analysis and interpretation
- **Technical Documentation**: Code and specification analysis
- **Safety Compliance**: Regulatory requirement verification
- **Quality Assurance**: Process validation and improvement

## Best Practices

### Input Optimization

1. **Image Preparation**
   - Ensure clear, high-contrast images
   - Crop to relevant problem area
   - Remove unnecessary background elements
   - Use consistent lighting conditions

2. **Screen Capture Optimization**
   - **Precise Selection**: Use corner handles for diagonal resizing, edge handles for proportional adjustment
   - **Area Focus**: Select only relevant regions to reduce processing time
   - **Minimum Size**: Maintain at least 50px dimensions for accurate analysis
   - **Boundary Awareness**: Selection automatically constrains to screen boundaries

3. **Problem Formulation**
   - Provide complete problem statements
   - Include all relevant parameters and constraints
   - Specify desired solution format or approach
   - Mention any specific requirements or preferences

4. **Context Provision**
   - Include relevant background information
   - Specify academic level or complexity
   - Mention previous attempts or known approaches
   - Provide any additional constraints or preferences

5. **API Configuration**
   - **Default API**: Use for immediate access without configuration
   - **Custom Endpoints**: Enter full URL including protocol (https://)
   - **Security**: No credentials stored - enhanced privacy protection
   - **Quick Switching**: Toggle between default and custom APIs as needed

### Solution Validation

1. **Accuracy Verification**
   - Cross-check with established methods
   - Verify intermediate calculations
   - Test boundary conditions and special cases
   - Compare with known solutions or results

2. **Completeness Assessment**
   - Ensure all problem aspects are addressed
   - Verify solution covers all required steps
   - Check for missing assumptions or conditions
   - Validate final answer against expectations

3. **Learning Integration**
   - Understand underlying principles and concepts
   - Identify key learning points from solutions
   - Apply similar approaches to related problems
   - Document insights for future reference

### Application-Specific Guidelines

#### Plant Identification
- **Image Quality**: Clear images showing key identification features
- **Multiple Angles**: Different views for comprehensive analysis
- **Scale Reference**: Include size reference when possible
- **Habitat Information**: Environmental context for accurate identification

#### Laboratory Safety
- **Comprehensive Views**: Show entire laboratory setup
- **Equipment Details**: Clear images of safety equipment and procedures
- **Hazard Identification**: Focus on potential risk areas
- **Regulatory Context**: Mention relevant safety standards

#### Academic Translation
- **Source Quality**: High-quality images of original text
- **Context Preservation**: Maintain academic style and terminology
- **Accuracy Verification**: Cross-check technical terms
- **Cultural Considerations**: Account for disciplinary conventions

## Troubleshooting

### Common Issues

**1. Poor Image Recognition**
- Improve image quality and lighting
- Ensure proper focus and contrast
- Crop to relevant problem area
- Try alternative camera angles

**2. Inaccurate Solutions**
- Provide more detailed problem context
- Select appropriate preset for problem type
- Include all relevant parameters
- Verify problem statement clarity

**3. Processing Delays**
- Check internet connection stability
- Reduce image file size if necessary
- Use text input for faster processing
- Try during lower network usage periods

**4. Camera Access Problems**
- Grant camera permissions in browser settings
- Check if other applications are using camera
- Verify camera hardware functionality
- Try different browser if issues persist

**5. Screen Capture Issues**
- **Selection Box Not Appearing**: Ensure screen sharing permissions are granted
- **Cannot Resize Selection**: Use corner handles for diagonal resizing, edge handles for proportional adjustment
- **Selection Stuck at Minimum Size**: Ensure selection area is at least 50px in both dimensions
- **Cannot Move Selection**: Click and drag the selection box itself (not handles)

**6. API Configuration Problems**
- **Default API Not Working**: Check internet connection and try refreshing the page
- **Custom API Connection Failed**: Verify URL format includes protocol (https://)
- **API Switching Issues**: Disable custom API first before enabling default API
- **Response Format Errors**: Ensure custom API returns compatible JSON format
- **Mock Response Displayed**: If you see a message explaining that a mock answer is used, deploy `/api/solve` or point the app to your custom endpoint

**7. Preset Selection Issues**
- **Preset Not Available**: All 18 presets should be visible in dropdown menu
- **Quick Command Not Working**: Type `/preset name` exactly as shown in preset list
- **Custom Mode Confusion**: Modifying question field automatically switches to Custom Mode

### Performance Optimization

**For Complex Problems**
- Break down complex problems into smaller components
- Use multiple specialized modes for different aspects
- Provide detailed context and constraints
- Consider step-by-step approach for multi-part problems

**For Educational Use**
- Start with simpler problems to build understanding
- Use multiple problem types for comprehensive learning
- Compare AI solutions with traditional methods
- Focus on understanding underlying principles

## Technical Support

If you encounter technical issues:

1. Check browser console for error messages
2. Verify image quality and input parameters
3. Ensure stable internet connection
4. Contact support with specific error details and problem examples

### Browser Compatibility
- **Chrome 60+**: Full camera, screen capture, and AI processing support
- **Firefox 55+**: Complete functionality with modern APIs including screen sharing
- **Safari 11+**: Camera access and AI integration (screen capture may have limitations)
- **Edge 79+**: Comprehensive support for all features including advanced selection

### Advanced Features Support
- **8-Directional Selection**: Chrome, Firefox, Edge (full support)
- **Screen Capture**: Chrome, Firefox, Edge (full support)
- **Default API**: All modern browsers
- **Quick Preset Commands**: All modern browsers

---
*Author: Liangchao Deng, Ph.D. Candidate, Shihezi University / CAS-CEMPS*  
*This tutorial applies to AI Solver v2.0*
*Enhanced with 18 professional presets, advanced selection tools, and flexible API configuration*
*Powered by Hunyuan AI for comprehensive problem-solving applications*
