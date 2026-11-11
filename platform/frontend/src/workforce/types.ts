export interface Metric { value: number | null; available: boolean; unit: string; source: string; note: string | null }
export interface PendingApproval {
  run_id: string; request_id: string; version: number; sent: string; expires_at: string;
  headline: string; restriction: string | null; duration: number;
  alert?: { trigger: string; observed: number; threshold: number; whatif?: { k: number; share_positive: number; estimated_gain: number } };
}
export interface Overview {
  jour: string; date: string; scenario: string; heure: string; phase: "pret" | "en_cours" | "pause" | "terminee" | "echec";
  etat: { code: string; libelle: string; niveau: string; texte: string };
  terminee: boolean; pause: boolean; vitesse: number; politique: string; fermeture: string;
  demande: PendingApproval | null; kpis: Record<string, Metric>; controle: Record<string, number | string> | null;
  intervention?: { run_id: string; id: string; debut: string; fin: string; restant: number; decisions: number; affected_decisions: number } | null;
  sante?: { etat: string; erreur: string | null; progression_bloquee: boolean; horloge_active: boolean; superviseur_degrade: boolean };
}
export interface ModelStatus { ok: boolean; error?: string; name?: string }
export interface Scenario { code: string; libelle: string; description: string; journee: string }
export interface Contact {
  date: string; jour: string; identifiant: string; client: string; heure: string; t: number;
  canal: string; type_demande: string; langue: string; priorite: string; score_ia: number;
  affectation: string; politique: string; statut: string; groupe: string; etat: string;
  resultat: string; attente_min: number | null; routages: string[]; traite_par: string;
}
export interface Decision {
  heure: string; t: number; action: string; decideur: string; confiance_ia: number;
  etat: Record<string, number>; autorisees: { action: string; autorisee: boolean; motif: string }[];
  explication?: string; proposition?: string | null; probas?: Record<string, number> | null;
}
export interface ContactEvent { heure: string; t: number; type: string; evenement: string; tentative?: number }
export interface ContactDetail {
  client: string; info: Record<string, string>; decisions: Decision[]; evenements: ContactEvent[];
  timeline?: { heure: string; texte: string; type: string }[]; etat_actuel?: string;
  resultat: string | null; note?: string; verite?: string | null; apparente?: boolean;
}
export interface ClientSummary {
  client: string; contacts: number; retours: number; premier: string; dernier: string; canal: string;
  type_demande: string; par_ia: number; par_conseiller: number; dernier_resultat: string; etat: string; motif_retour: string | null;
}
export interface HistoryDay { date: string; jour: string; scenario: string; archivee: boolean }
export interface ClientHistory extends ClientSummary { appels: (Contact & ContactDetail)[] }
export interface History { contacts: Contact[]; clients: ClientSummary[]; retours: Record<string, number | null | Record<string, number>> }
export type ChartRow = { heure: string } & Record<string, number | string | null>;
export interface Charts {
  demandes: ChartRow[]; charge: ChartRow[]; qualite: ChartRow[];
  routage: { action: string; libelle: string; decisions: number; part: number | null }[];
}
export interface Attention { en_attente: Contact[]; rappels: Contact[]; validation: PendingApproval | null; incidents: { texte: string; n: number }[] }
export interface Supervision {
  journees?: { jour: string; scenario: string; terminee: boolean; contacts_recus: number; decisions_ia: number | null; abandon: number | null; resolution: number | null; attente_moyenne: number | null; echecs_ia: number }[];
  historique?: { jour: string; heure: string; evenement: string; detail: string }[];
  alertes: { jour: string; heure: string; alerte: string; valeur: number; seuil: number; decision: string; suite: string | null }[];
  transitions: { jour: string; heure: string; de: string; vers: string; motif: string }[];
  interventions: { jour: string; envoyee: string; proposition: string; reponse: string; a: string | null }[];
  secours: { jour: string; heure: string; evenement: string }[];
}
export interface AiAlert { taux: number | null; seuil: number; regle: string; etat: string; texte_echecs?: string; fenetre?: string }
export interface DaySummary {
  jour: string; scenario: string; terminee: boolean; heure: string; demandes_recues: number; termines: number;
  non_termines: number; resolues: number; dont_ia_apparentes: number; abandonnees: number; autres_echecs: number;
  decisions: number; decisions_ia: number | null; charge_humaine: number; attente_moyenne: number | null;
  attente_max: number | null; echecs_ia: number; alertes: number; rappels: Record<string, number>;
  liste_alertes: Supervision["alertes"]; interventions: Supervision["interventions"];
}
