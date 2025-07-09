import React from 'react';
import {
  makeStyles,
  Title1,
  Body1,
  Badge,
} from '@fluentui/react-components';
import { ClockRegular } from '@fluentui/react-icons';

const useStyles = makeStyles({
  header: {
    display: 'flex',
    alignItems: 'center',
    padding: '16px 20px',
    backgroundColor: '#ffffff',
    borderBottom: '1px solid #e0e0e0',
    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  icon: {
    fontSize: '24px',
    color: '#0078d4',
  },
  title: {
    color: '#323130',
    fontWeight: '600',
  },
  subtitle: {
    color: '#605e5c',
    marginLeft: '16px',
  },
  badge: {
    marginLeft: 'auto',
  },
});

export const Header: React.FC = () => {
  const styles = useStyles();

  return (
    <header className={styles.header}>
      <div className={styles.logo}>
        <ClockRegular className={styles.icon} />
        <Title1 className={styles.title}>TimeCraft</Title1>
      </div>
      <Body1 className={styles.subtitle}>
        Time Series Generation for Real-World Applications
      </Body1>
      <div className={styles.badge}>
        <Badge appearance="filled" color="brand">
          v1.0.0
        </Badge>
      </div>
    </header>
  );
};