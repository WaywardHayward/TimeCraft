# TimeCraft Frontend Documentation

## Overview

TimeCraft now includes a modern React frontend built with Microsoft's Fluent UI System, providing a user-friendly web interface for all TimeCraft functionality.

## Features

### 🎨 Modern UI with Fluent UI
- Clean, professional interface using Microsoft's Fluent UI components
- Responsive design that works on desktop and mobile devices
- Consistent theming and visual design language
- Tabbed navigation for easy access to different features

### 📊 Upload & Analyze
- **Drag-and-drop file upload** for CSV time series data
- **Configurable parameters**:
  - Dataset name customization
  - Prediction length adjustment
  - LLM optimization toggle
  - OpenAI API configuration (Azure OpenAI supported)
- **Real-time feedback** with loading indicators and error handling

### ✨ Text Refinement
- **Multi-agent text refinement** using advanced LLM approaches
- **Configurable refinement parameters**:
  - Team iterations (1-10)
  - Global iterations (1-5)
- **OpenAI integration** with support for Azure OpenAI
- **Side-by-side comparison** of original vs refined text

### 🔍 System Status
- **Real-time component monitoring** showing availability of:
  - TimeCraft Core
  - BRIDGE Components
  - Text-to-TimeSeries functionality
  - TimeDP Components
  - TarDiff Components
  - Pandas Library
  - API Server
- **Health checks** with automatic status updates
- **API information** including version and documentation links

## Getting Started

### Prerequisites
- Node.js 16+ installed
- Python 3.8+ with FastAPI dependencies
- TimeCraft backend components (optional for full functionality)

### Running the Frontend

1. **Install dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Start development server**:
   ```bash
   npm start
   ```

3. **Build for production**:
   ```bash
   npm run build
   ```

### Running the Full Stack

1. **Start the FastAPI backend**:
   ```bash
   python api_server.py
   ```

2. **Access the application**:
   - Open your browser to `http://localhost:8080`
   - The React frontend will be served automatically
   - API endpoints are available at `http://localhost:8080/api/*`

## Architecture

### Frontend Stack
- **React 18** with TypeScript for type safety
- **Microsoft Fluent UI** for consistent, modern UI components
- **Axios** for API communication
- **Create React App** for build tooling

### API Integration
- **RESTful API** communication with the FastAPI backend
- **Automatic proxy** configuration for development
- **Error handling** with user-friendly messages
- **File upload** support for CSV data

### Component Structure
```
src/
├── components/           # React UI components
│   ├── Header.tsx       # Application header with branding
│   ├── FileUpload.tsx   # CSV upload and analysis
│   ├── TextRefinement.tsx # Multi-agent text refinement
│   └── SystemStatus.tsx # Component status monitoring
├── services/            # API integration
│   └── ApiClient.ts     # Centralized API communication
└── App.tsx             # Main application with routing
```

## Configuration

### Environment Variables
- `REACT_APP_API_URL` - API base URL (defaults to relative paths)

### Backend Configuration
The FastAPI backend automatically serves the React frontend when built. Static files are served from the `/frontend/build` directory.

## Development

### Adding New Features
1. Create new components in `src/components/`
2. Add API methods to `src/services/ApiClient.ts`
3. Update the main `App.tsx` to include new tabs/routes
4. Follow Fluent UI design patterns for consistency

### Styling Guidelines
- Use Fluent UI components wherever possible
- Follow the established color scheme and spacing
- Maintain responsive design principles
- Use TypeScript for type safety

## Troubleshooting

### Common Issues

1. **Frontend not loading**: Ensure `npm run build` has been executed
2. **API errors**: Verify the FastAPI backend is running on port 8080
3. **Missing components**: Check the System Status tab for component availability
4. **File upload failures**: Ensure files are in CSV format and under size limits

### Development Tips

1. **Use browser dev tools** to monitor network requests
2. **Check console logs** for detailed error messages
3. **Use the System Status tab** to verify backend connectivity
4. **Test with sample CSV files** before using production data

## Screenshots

The frontend includes three main sections:

1. **Upload & Analyze**: Clean file upload interface with configuration options
2. **Text Refinement**: Multi-agent text processing with parameter controls
3. **System Status**: Real-time monitoring of all TimeCraft components

## Support

For issues or questions about the frontend:
1. Check the System Status tab for component availability
2. Review browser console logs for errors
3. Refer to the API documentation at `/swagger`
4. Create an issue in the TimeCraft repository