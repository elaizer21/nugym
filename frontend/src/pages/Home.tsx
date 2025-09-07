import React from "react";
import { Container, Typography, Box } from "@mui/material";

const Home = () => (
  <Container>
    <Box my={6}>
      <Typography variant="h3" color="primary" gutterBottom>
        Welcome to FitSync Pro
      </Typography>
      <Typography variant="h6">
        Track your fitness. Sync your body scans, food, and workouts. Predict your progress and manage subscriptions—all in one place!
      </Typography>
    </Box>
  </Container>
);

export default Home;