import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { AuthProvider } from "@/auth";
import { App } from "@/App";
import { I18nProvider } from "@/i18n";
import "@/index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <I18nProvider><AuthProvider>
        <App />
      </AuthProvider></I18nProvider>
    </BrowserRouter>
  </StrictMode>,
);
