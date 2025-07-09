import React, { useState, useEffect } from 'react';
import {
  FluentProvider,
  webLightTheme,
  makeStyles,
  Tab,
  TabList,
  TabValue,
} from '@fluentui/react-components';
import { Header } from './components/Header';
import { FileUpload } from './components/FileUpload';
import { TextRefinement } from './components/TextRefinement';
import { SystemStatus } from './components/SystemStatus';
import { ApiClient } from './services/ApiClient';

const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    backgroundColor: '#fafafa',
  },
  content: {
    flex: 1,
    padding: '20px',
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
  },
  tabContent: {
    flex: 1,
    padding: '20px 0',
  },
});

function App() {
  const styles = useStyles();
  const [selectedTab, setSelectedTab] = useState<TabValue>('upload');
  const [systemStatus, setSystemStatus] = useState<any>(null);

  useEffect(() => {
    const fetchSystemStatus = async () => {
      try {
        const status = await ApiClient.getSystemStatus();
        setSystemStatus(status);
      } catch (error) {
        console.error('Failed to fetch system status:', error);
      }
    };

    fetchSystemStatus();
  }, []);

  const handleTabSelect = (event: any, data: { value: TabValue }) => {
    setSelectedTab(data.value);
  };

  return (
    <FluentProvider theme={webLightTheme}>
      <div className={styles.root}>
        <Header />
        <div className={styles.content}>
          <TabList selectedValue={selectedTab} onTabSelect={handleTabSelect}>
            <Tab value="upload">Upload & Analyze</Tab>
            <Tab value="refine">Text Refinement</Tab>
            <Tab value="status">System Status</Tab>
          </TabList>
          
          <div className={styles.tabContent}>
            {selectedTab === 'upload' && <FileUpload />}
            {selectedTab === 'refine' && <TextRefinement />}
            {selectedTab === 'status' && <SystemStatus status={systemStatus} />}
          </div>
        </div>
      </div>
    </FluentProvider>
  );
}

export default App;
