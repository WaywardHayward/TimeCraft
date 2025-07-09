import React, { useState, useEffect } from 'react';
import {
  makeStyles,
  Card,
  CardHeader,
  Button,
  Badge,
  Body1,
  Title3,
  Divider,
  MessageBar,
} from '@fluentui/react-components';
import {
  CheckmarkCircleRegular,
  ErrorCircleRegular,
  WarningRegular,
  ArrowClockwiseRegular,
} from '@fluentui/react-icons';
import { ApiClient } from '../services/ApiClient';

const useStyles = makeStyles({
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
  },
  statusCard: {
    width: '100%',
    maxWidth: '800px',
  },
  statusGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: '16px',
    padding: '16px',
  },
  statusItem: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '12px',
    border: '1px solid #e0e0e0',
    borderRadius: '4px',
    backgroundColor: '#f9f9f9',
  },
  statusLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  refreshButton: {
    alignSelf: 'flex-start',
  },
  apiInfo: {
    padding: '16px',
    backgroundColor: '#f3f9ff',
    borderRadius: '4px',
    marginTop: '16px',
  },
});

interface SystemStatusProps {
  status: any;
}

export const SystemStatus: React.FC<SystemStatusProps> = ({ status: initialStatus }) => {
  const styles = useStyles();
  const [status, setStatus] = useState(initialStatus);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<any>(null);

  useEffect(() => {
    setStatus(initialStatus);
  }, [initialStatus]);

  useEffect(() => {
    fetchHealth();
  }, []);

  const fetchHealth = async () => {
    try {
      const healthData = await ApiClient.getHealth();
      setHealth(healthData);
    } catch (err: any) {
      console.error('Failed to fetch health:', err);
    }
  };

  const refreshStatus = async () => {
    setLoading(true);
    setError(null);

    try {
      const [newStatus, newHealth] = await Promise.all([
        ApiClient.getSystemStatus(),
        ApiClient.getHealth(),
      ]);
      setStatus(newStatus);
      setHealth(newHealth);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to refresh status');
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (available: boolean) => {
    return available ? (
      <CheckmarkCircleRegular style={{ color: '#107c10' }} />
    ) : (
      <ErrorCircleRegular style={{ color: '#d13438' }} />
    );
  };

  const getStatusBadge = (available: boolean) => {
    return (
      <Badge 
        appearance="filled" 
        color={available ? 'success' : 'danger'}
      >
        {available ? 'Available' : 'Unavailable'}
      </Badge>
    );
  };

  const components = [
    { name: 'TimeCraft Core', key: 'timecraft_available' },
    { name: 'BRIDGE Components', key: 'bridge_available' },
    { name: 'Text-to-TimeSeries', key: 'bridge_text_to_ts' },
    { name: 'TimeDP Components', key: 'timedp_available' },
    { name: 'TarDiff Components', key: 'tardiff_available' },
    { name: 'Pandas Library', key: 'pandas_available' },
    { name: 'API Server', key: 'api_server' },
  ];

  return (
    <div className={styles.container}>
      <Card className={styles.statusCard}>
        <CardHeader
          header={<Title3>System Status</Title3>}
          description="Current status of TimeCraft components and dependencies"
          action={
            <Button
              className={styles.refreshButton}
              icon={<ArrowClockwiseRegular />}
              onClick={refreshStatus}
              disabled={loading}
            >
              {loading ? 'Refreshing...' : 'Refresh'}
            </Button>
          }
        />

        {error && (
          <div style={{ padding: '0 16px' }}>
            <MessageBar intent="error">
              {error}
            </MessageBar>
          </div>
        )}

        {health && (
          <div style={{ padding: '0 16px' }}>
            <MessageBar 
              intent={health.status === 'healthy' ? 'success' : 'warning'}
            >
              API Health: {health.status} - {health.message}
            </MessageBar>
          </div>
        )}

        <div className={styles.statusGrid}>
          {components.map((component) => {
            const isAvailable = status?.[component.key] === true || status?.[component.key] === 'true';
            return (
              <div key={component.key} className={styles.statusItem}>
                <div className={styles.statusLabel}>
                  {getStatusIcon(isAvailable)}
                  <Body1>{component.name}</Body1>
                </div>
                {getStatusBadge(isAvailable)}
              </div>
            );
          })}
        </div>

        <Divider />

        {status && (
          <div className={styles.apiInfo}>
            <Title3>API Information</Title3>
            <Body1>
              <strong>Version:</strong> {status.version || '1.0.0'}
            </Body1>
            <Body1>
              <strong>Documentation:</strong> <a href="/swagger" target="_blank" rel="noopener noreferrer">/swagger</a>
            </Body1>
            <Body1>
              <strong>Base URL:</strong> {process.env.REACT_APP_API_URL || 'http://localhost:8080'}
            </Body1>
          </div>
        )}
      </Card>

      {status?.demo_mode && (
        <MessageBar intent="warning">
          <WarningRegular style={{ marginRight: '8px' }} />
          Running in demo mode. Some components are not available. Please check the installation and configuration.
        </MessageBar>
      )}
    </div>
  );
};