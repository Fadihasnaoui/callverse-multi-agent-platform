/** Shared types matching the backend API responses. */

export interface Business {
  id: string;
  name: string;
  industry: string;
  brand_voice: string;
  status: string;
  created_at: string;
  connected: string[];
}

export interface Overview {
  content: Record<string, number>;
  needs_media: number;
  upcoming: UpcomingItem[];
  scheduled_total?: number;
  leads_new_30d: number;
  sales_30d: { count: number; revenue: number };
  connected: string[];
  performance_30d: {
    availability?: {reach:boolean;impressions:boolean;engagement:boolean};
    definition?: string;
    reach: number; impressions: number; engagement: number; engagement_rate: number | null;
    saves: number; video_views: number; last_synced_at: string | null; posts_with_metrics: number;
  };
  content_mix: Record<string, number>;
  platform_posts: Record<string, { agent: number; page_import: number; total: number }>;
  page_kpis: Record<string, Record<string, number | string | null>>;
  platform_performance: Record<string, {
    posts: number; total_likes: number; total_comments: number;
    total_reactions_shares: number; total_engagement: number;
    avg_engagement_per_post: number; engagement_rate: number | null;
  }>;
  category_distribution: Record<string, number>;
  category_performance: Record<string, {
    posts: number; avg_likes: number; avg_comments: number; avg_engagement: number;
  }>;
  strategy_mix: Record<string, number>;
  posting_cadence: Record<string, { total_90d: number; per_week: number }>;
  weekly_performance: { week: string; platform: string; engagement: number }[];
  latest_insight: { summary: string | null; recommendations: string[]; as_of: string } | null;
  latest_audit: { posts_analyzed: number; as_of: string } | null;
  strategist: {
    platform: string;
    suggested_category: string;
    suggested_strategy_layer: string;
    suggested_format: string;
    suggested_funnel_stage: string;
    reasoning: string;
    platform_playbook: Record<string, string>;
  } | null;
}

export interface CampaignSimulation {
  mode: "simulation";
  label: string;
  numbers: {
    reach: number; impressions: number; engagement: number; engagement_rate: number;
    leads: number; sales: number; revenue: number; planned: number; approved: number;
    scheduled: number; published: number;
  };
  content_mix: Record<string, number>;
  platform_mix: Record<string, number>;
  weekly: { label: string; facebook: number; instagram: number }[];
  learning: { confidence: "simulated"; summary: string; recommendations: string[] };
}

export interface UpcomingItem {
  id: string;
  type: string;
  platform: string;
  caption: string;
  scheduled_for: string;
}

export interface ContentItem {
  content_version?: number;
  approved_version?: number | null;
  proposed_schedule?: string | null;
  id: string;
  type: string;
  platform: string;
  status: string;
  caption: string;
  hashtags: string[];
  scheduled_for: string | null;
  created_at: string;
  photo_urls: string[];
  video_url: string | null;
  publish_error: string | null;
  meta_post_id: string | null;
  avatar_text: string | null;
  avatar_language: string | null;
  avatar_transcript_status: string | null;
  avatar_job_id: string | null;
  visual_brief: string | null;
  music_url: string | null;
}
