import React, { useState, useCallback } from 'react';
import {
  makeStyles,
  Card,
  CardHeader,
  CardFooter,
  Button,
  Input,
  Label,
  Textarea,
  Spinner,
  MessageBar,
  Switch,
  Body1,
  Title3,
  Divider,
} from '@fluentui/react-components';
import {
  DocumentArrowUpRegular,
  PlayRegular,
} from '@fluentui/react-icons';
import { ApiClient } from '../services/ApiClient';

const useStyles = makeStyles({
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
  },
  uploadCard: {
    width: '100%',
    maxWidth: '800px',
  },
  uploadArea: {
    border: '2px dashed #c8c6c4',
    borderRadius: '4px',
    padding: '40px',
    textAlign: 'center' as const,
    cursor: 'pointer',
  },
  uploadAreaActive: {
    border: '2px dashed #0078d4',
    borderRadius: '4px',
    padding: '40px',
    textAlign: 'center' as const,
    cursor: 'pointer',
    backgroundColor: '#f3f9ff',
  },
  formRow: {
    display: 'flex',
    gap: '16px',
    alignItems: 'end',
  },
  formField: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
    flex: 1,
  },
  results: {
    marginTop: '20px',
  },
  resultCard: {
    width: '100%',
  },
  loadingContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '20px',
  },
});

export const FileUpload: React.FC = () => {
  const styles = useStyles();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState({
    datasetName: 'uploaded_dataset',
    predictionLength: 168,
    llmOptimize: false,
    openaiApiBase: '',
    openaiApiVersion: '',
    openaiApiType: 'azure',
  });

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      setSelectedFile(files[0]);
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      setSelectedFile(files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const result = await ApiClient.uploadFile(selectedFile, {
        datasetName: formData.datasetName,
        predictionLength: formData.predictionLength,
        llmOptimize: formData.llmOptimize,
        openaiApiBase: formData.openaiApiBase || undefined,
        openaiApiVersion: formData.openaiApiVersion || undefined,
        openaiApiType: formData.openaiApiType || undefined,
      });
      setResults(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <Card className={styles.uploadCard}>
        <CardHeader
          header={<Title3>Upload Time Series Data</Title3>}
          description="Upload CSV files to generate textual descriptions"
        />
        
        <div
          className={`${styles.uploadArea} ${dragActive ? styles.uploadAreaActive : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-input')?.click()}
        >
          <DocumentArrowUpRegular style={{ fontSize: '48px', color: '#0078d4' }} />
          <Body1>
            {selectedFile ? selectedFile.name : 'Drop CSV file here or click to browse'}
          </Body1>
          <input
            id="file-input"
            type="file"
            accept=".csv"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
        </div>

        <Divider />

        <div className={styles.formRow}>
          <div className={styles.formField}>
            <Label htmlFor="dataset-name">Dataset Name</Label>
            <Input
              id="dataset-name"
              value={formData.datasetName}
              onChange={(e) => setFormData(prev => ({ ...prev, datasetName: e.target.value }))}
            />
          </div>
          <div className={styles.formField}>
            <Label htmlFor="prediction-length">Prediction Length</Label>
            <Input
              id="prediction-length"
              type="number"
              value={formData.predictionLength.toString()}
              onChange={(e) => setFormData(prev => ({ ...prev, predictionLength: parseInt(e.target.value) || 168 }))}
            />
          </div>
          <div className={styles.formField}>
            <Label htmlFor="llm-optimize">LLM Optimize</Label>
            <Switch
              id="llm-optimize"
              checked={formData.llmOptimize}
              onChange={(e) => setFormData(prev => ({ ...prev, llmOptimize: e.currentTarget.checked }))}
            />
          </div>
        </div>

        {formData.llmOptimize && (
          <>
            <Divider />
            <div className={styles.formRow}>
              <div className={styles.formField}>
                <Label htmlFor="openai-base">OpenAI API Base</Label>
                <Input
                  id="openai-base"
                  placeholder="https://your-resource.openai.azure.com/"
                  value={formData.openaiApiBase}
                  onChange={(e) => setFormData(prev => ({ ...prev, openaiApiBase: e.target.value }))}
                />
              </div>
              <div className={styles.formField}>
                <Label htmlFor="openai-version">API Version</Label>
                <Input
                  id="openai-version"
                  placeholder="2024-02-15-preview"
                  value={formData.openaiApiVersion}
                  onChange={(e) => setFormData(prev => ({ ...prev, openaiApiVersion: e.target.value }))}
                />
              </div>
            </div>
          </>
        )}

        <CardFooter>
          <Button
            appearance="primary"
            icon={<PlayRegular />}
            disabled={!selectedFile || loading}
            onClick={handleUpload}
          >
            {loading ? 'Processing...' : 'Generate Descriptions'}
          </Button>
        </CardFooter>
      </Card>

      {loading && (
        <div className={styles.loadingContainer}>
          <Spinner size="small" />
          <Body1>Processing your file...</Body1>
        </div>
      )}

      {error && (
        <MessageBar intent="error">
          {error}
        </MessageBar>
      )}

      {results && (
        <Card className={styles.resultCard}>
          <CardHeader
            header={<Title3>Results</Title3>}
            description="Generated descriptions from your time series data"
          />
          <div style={{ padding: '16px' }}>
            <Label>Status: {results.status}</Label>
            <Textarea
              value={JSON.stringify(results, null, 2)}
              readOnly
              rows={10}
              style={{ marginTop: '8px' }}
            />
          </div>
        </Card>
      )}
    </div>
  );
};