import React from "react";
import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Profile from "./pages/Profile";
import BodyScan from "./pages/BodyScan";
import FoodLog from "./pages/FoodLog";
import ExerciseLog from "./pages/ExerciseLog";
import Prediction from "./pages/Prediction";
import Subscription from "./pages/Subscription";
import EmailVerify from "./pages/EmailVerify";
import PasswordReset from "./pages/PasswordReset";
import NavBar from "./components/NavBar";
import ProtectedRoute from "./components/ProtectedRoute";

function App() {
  return (
    <> 
      <NavBar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/verify-email" element={<EmailVerify />} />
        <Route path="/reset-password" element={<PasswordReset />} />
        <Route element={<ProtectedRoute />}> 
          <Route path="/profile" element={<Profile />} />
          <Route path="/body-scan" element={<BodyScan />} />
          <Route path="/food-log" element={<FoodLog />} />
          <Route path="/exercise-log" element={<ExerciseLog />} />
          <Route path="/prediction" element={<Prediction />} />
          <Route path="/subscription" element={<Subscription />} />
        </Route>
      </Routes>
    </>
  );
}

export default App;