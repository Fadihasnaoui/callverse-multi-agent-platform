import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from "react";
import { api, ApiError } from "@/api";

interface AuthState {
  checked: boolean;
  loggedIn: boolean;
  authRequired: boolean;
}

interface AuthCtx extends AuthState {
  login: (password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const Ctx = createContext<AuthCtx | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({ checked: false, loggedIn: false, authRequired: true });

  const check = useCallback(async () => {
    try {
      const data = await api.get<{ authenticated: boolean; auth_required: boolean }>("/auth/me");
      setState({ checked: true, loggedIn: data.authenticated, authRequired: data.auth_required });
    } catch {
      setState({ checked: true, loggedIn: false, authRequired: true });
    }
  }, []);

  useEffect(() => { check(); }, [check]);

  const login = useCallback(async (password: string) => {
    try {
      await api.post("/auth/login", { password });
      await check();
    } catch (e) {
      if (e instanceof ApiError && e.status === 429) throw new Error("Too many attempts — wait a few minutes");
      throw new Error("Wrong password");
    }
  }, [check]);

  const logout = useCallback(async () => {
    await api.post("/auth/logout");
    setState((s) => ({ ...s, loggedIn: false }));
  }, []);

  return <Ctx value={{ ...state, login, logout }}>{children}</Ctx>;
}

export function useAuth() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useAuth must be inside AuthProvider");
  return ctx;
}
