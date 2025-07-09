import React, { useState } from 'react';
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
  Body1,
  Title3,
  Divider,
} from '@fluentui/react-components';
import {
  EditRegular,
} from '@fluentui/react-icons';
import { ApiClient } from '../services/ApiClient';

const useStyles = makeStyles({
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
  },
  refinementCard: {
    width: '100%',
    maxWidth: '800px',
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
  textareaField: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
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
  comparisonContainer: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '16px',
    marginTop: '16px',
  },
  textColumn: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
});

export const TextRefinement: React.FC = () => {
  const styles = useStyles();
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState({
    initialText: '',
    teamIterations: 3,
    globalIterations: 2,
    openaiApiBase: '',
    openaiApiVersion: '',
    openaiApiType: 'azure',
    openaiApiKey: '',
  });

  const handleRefine = async () => {
    if (!formData.initialText.trim()) {
      setError('Please enter text to refine');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const result = await ApiClient.refineText({
        initial_text: formData.initialText,
        team_iterations: formData.teamIterations,
        global_iterations: formData.globalIterations,
        openai_api_base: formData.openaiApiBase || undefined,
        openai_api_version: formData.openaiApiVersion || undefined,
        openai_api_type: formData.openaiApiType || undefined,
        openai_api_key: formData.openaiApiKey || undefined,
      });
      setResults(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Text refinement failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <Card className={styles.refinementCard}>
        <CardHeader
          header={<Title3>Text Refinement</Title3>}
          description="Use multi-agent approach to refine textual descriptions"
        />
        
        <div className={styles.textareaField}>
          <Label htmlFor="initial-text">Initial Text</Label>
          <Textarea
            id="initial-text"
            placeholder="Enter the text you want to refine..."
            value={formData.initialText}
            onChange={(e) => setFormData(prev => ({ ...prev, initialText: e.target.value }))}
            rows={6}
          />
        </div>

        <Divider />

        <div className={styles.formRow}>
          <div className={styles.formField}>
            <Label htmlFor="team-iterations">Team Iterations</Label>
            <Input
              id="team-iterations"
              type="number"
              value={formData.teamIterations.toString()}
              onChange={(e) => setFormData(prev => ({ ...prev, teamIterations: parseInt(e.target.value) || 3 }))}
              min={1}
              max={10}
            />
          </div>
          <div className={styles.formField}>
            <Label htmlFor="global-iterations">Global Iterations</Label>
            <Input
              id="global-iterations"
              type="number"
              value={formData.globalIterations.toString()}
              onChange={(e) => setFormData(prev => ({ ...prev, globalIterations: parseInt(e.target.value) || 2 }))}
              min={1}
              max={5}
            />
          </div>
        </div>

        <Divider />

        <div className={styles.formRow}>
          <div className={styles.formField}>
            <Label htmlFor="openai-base-refine">OpenAI API Base</Label>
            <Input
              id="openai-base-refine"
              placeholder="https://your-resource.openai.azure.com/"
              value={formData.openaiApiBase}
              onChange={(e) => setFormData(prev => ({ ...prev, openaiApiBase: e.target.value }))}
            />
          </div>
          <div className={styles.formField}>
            <Label htmlFor="openai-version-refine">API Version</Label>
            <Input
              id="openai-version-refine"
              placeholder="2024-02-15-preview"
              value={formData.openaiApiVersion}
              onChange={(e) => setFormData(prev => ({ ...prev, openaiApiVersion: e.target.value }))}
            />
          </div>
        </div>

        <div className={styles.formField}>
          <Label htmlFor="openai-key">OpenAI API Key</Label>
          <Input
            id="openai-key"
            type="password"
            placeholder="Your OpenAI API key"
            value={formData.openaiApiKey}
            onChange={(e) => setFormData(prev => ({ ...prev, openaiApiKey: e.target.value }))}
          />
        </div>

        <CardFooter>
          <Button
            appearance="primary"
            icon={<EditRegular />}
            disabled={!formData.initialText.trim() || loading}
            onClick={handleRefine}
          >
            {loading ? 'Refining...' : 'Refine Text'}
          </Button>
        </CardFooter>
      </Card>

      {loading && (
        <div className={styles.loadingContainer}>
          <Spinner size="small" />
          <Body1>Refining your text using multi-agent approach...</Body1>
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
            header={<Title3>Refinement Results</Title3>}
            description="Comparison between original and refined text"
          />
          <div style={{ padding: '16px' }}>
            <Label>Status: {results.status}</Label>
            
            {results.original_text && results.refined_text && (
              <div className={styles.comparisonContainer}>
                <div className={styles.textColumn}>
                  <Label>Original Text</Label>
                  <Textarea
                    value={results.original_text}
                    readOnly
                    rows={8}
                  />
                </div>
                <div className={styles.textColumn}>
                  <Label>Refined Text</Label>
                  <Textarea
                    value={results.refined_text}
                    readOnly
                    rows={8}
                  />
                </div>
              </div>
            )}
            
            {results.refinement_logs && (
              <div style={{ marginTop: '16px' }}>
                <Label>Refinement Logs</Label>
                <Textarea
                  value={JSON.stringify(results.refinement_logs, null, 2)}
                  readOnly
                  rows={6}
                  style={{ marginTop: '8px' }}
                />
              </div>
            )}
          </div>
        </Card>
      )}
    </div>
  );
};