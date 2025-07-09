import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || '';

class ApiClientClass {
  private axiosInstance;

  constructor() {
    this.axiosInstance = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  async getSystemStatus() {
    const response = await this.axiosInstance.get('/api/status');
    return response.data;
  }

  async getHealth() {
    const response = await this.axiosInstance.get('/api/health');
    return response.data;
  }

  async uploadFile(file: File, options: {
    datasetName?: string;
    predictionLength?: number;
    llmOptimize?: boolean;
    openaiApiBase?: string;
    openaiApiVersion?: string;
    openaiApiType?: string;
  } = {}) {
    const formData = new FormData();
    formData.append('file', file);
    
    if (options.datasetName) formData.append('dataset_name', options.datasetName);
    if (options.predictionLength) formData.append('prediction_length', options.predictionLength.toString());
    if (options.llmOptimize) formData.append('llm_optimize', options.llmOptimize.toString());
    if (options.openaiApiBase) formData.append('openai_api_base', options.openaiApiBase);
    if (options.openaiApiVersion) formData.append('openai_api_version', options.openaiApiVersion);
    if (options.openaiApiType) formData.append('openai_api_type', options.openaiApiType);

    const response = await this.axiosInstance.post('/api/generate-description', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async refineText(data: {
    initial_text: string;
    team_iterations?: number;
    global_iterations?: number;
    openai_api_base?: string;
    openai_api_version?: string;
    openai_api_type?: string;
    openai_api_key?: string;
  }) {
    const response = await this.axiosInstance.post('/api/refine-text', data);
    return response.data;
  }

  async analyzeCSV(file: File) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.axiosInstance.post('/api/analyze-csv', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async generateTimeSeries(data: {
    text_description: string;
    series_length?: number;
    openai_api_base?: string;
    openai_api_version?: string;
    openai_api_type?: string;
    openai_api_key?: string;
  }) {
    const response = await this.axiosInstance.post('/api/generate-timeseries-from-text', data);
    return response.data;
  }
}

export const ApiClient = new ApiClientClass();