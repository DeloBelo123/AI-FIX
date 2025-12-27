import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_KEY!

if (!supabaseUrl) throw new Error("No NEXT_PUBLIC_SUPABASE_URL in env")
if (!supabaseServiceKey) throw new Error("No SUPABASE_SERVICE_ROLE_KEY or NEXT_PUBLIC_SUPABASE_KEY in env")

export const supabase = createClient(supabaseUrl, supabaseServiceKey)