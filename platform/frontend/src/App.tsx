import { Routes, Route, Navigate } from "react-router-dom";
import { useAuth } from "@/auth";
import { Login } from "@/pages/Login";
import { Customers } from "@/pages/Customers";
import { CustomerShell } from "@/pages/CustomerShell";
import { Dashboard } from "@/pages/Dashboard";
import { Content } from "@/pages/Content";
import { Calendar } from "@/pages/Calendar";
import { Leads } from "@/pages/Leads";
import { Report } from "@/pages/Report";
import { Audit } from "@/pages/Audit";
import { Settings } from "@/pages/Settings";
import { AIStats } from "@/pages/AIStats";
import { AgentWorkflow } from "@/pages/AgentWorkflow";
import { Costs } from "@/pages/Costs";
import { Catalogue } from "@/pages/Catalogue";
import { AgentHub } from "@/pages/AgentHub";
import { WorkforceWorkspace } from "@/pages/Workforce";
import { DigitalTwin } from "@/pages/DigitalTwin";
import { QualityReports } from "@/pages/QualityReports";
import { AgentPortal } from "@/pages/AgentPortal";

export function App() {
  const { checked, loggedIn } = useAuth();

  if (!checked) return null; // waiting for /auth/me

  if (!loggedIn) return <Login />;

  return (
    <Routes>
      <Route path="/" element={<AgentHub />} />
      <Route path="/marketing" element={<Customers />} />
      <Route path="/workforce" element={<WorkforceWorkspace />} />
      <Route path="/digital-twin" element={<DigitalTwin />} />
      <Route path="/customer-advisor" element={<AgentPortal title="Customer Advisor / Amira" label="Amira original dashboard" source="http://127.0.0.1:5177" description="Use Amira independently; her original dashboard and conversation workspace remain available." />} />
      <Route path="/client-simulator" element={<AgentPortal title="Client Simulator / Jasmin" label="Jasmin original dashboard" source="http://127.0.0.1:8012" description="Create and inspect customer personas and scenarios independently before using them in the Digital Twin." />} />
      <Route path="/quality-analyst" element={<QualityReports />} />
      <Route path="/quality-analyst/original" element={<AgentPortal title="Quality Analyst" label="Quality Analyst original dashboard" source="http://127.0.0.1:5174" description="Review saved conversations independently. The Digital Twin also sends its shared transcript here for evidence-based analysis." />} />
      <Route path="/workforce-lab" element={<AgentPortal title="Workforce Manager Lab" label="Workforce original dashboard" source="http://127.0.0.1:8514" description="Keep the original Workforce research simulation separate from real Digital Twin operational sessions." />} />
      <Route path="/ai-stats" element={<AIStats />} />
      <Route path="/c/:businessId" element={<CustomerShell />}>
        <Route index element={<Dashboard />} />
        <Route path="content" element={<Content />} />
        <Route path="calendar" element={<Calendar />} />
        <Route path="leads" element={<Leads />} />
        <Route path="report" element={<Report />} />
        <Route path="audit" element={<Audit />} />
        <Route path="agent-workflow" element={<AgentWorkflow />} />
        <Route path="costs" element={<Costs />} />
        <Route path="catalogue" element={<Catalogue />} />
        <Route path="settings" element={<Settings />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
