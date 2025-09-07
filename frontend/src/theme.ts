import { createTheme } from "@mui/material/styles";

const theme = createTheme({
  palette: {
    mode: "light",
    primary: { main: "#008080" },
    secondary: { main: "#ffe082" },
    background: { default: "#f6f9fc" },
  },
  typography: {
    fontFamily: "Montserrat, Arial, sans-serif",
  },
});

export default theme;