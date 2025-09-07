import { useState, useEffect } from "react";
import axios from "axios";

const API = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

export default function useAuth() {
  const [user, setUser] = useState(null);

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (token) {
      axios.get(`${API}/users/me`, {
        headers: { Authorization: `Bearer ${token}` },
      }).then(res => setUser(res.data)).catch(() => setUser(null));
    }
  }, []);

  const logout = () => {
    localStorage.removeItem("token");
    setUser(null);
  }; 

  return { user, logout };
}