import { createClient, type Provider } from "@supabase/supabase-js";
import { SupabaseTable } from "./server";
import axios from "axios";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_KEY!

if (!supabaseUrl) throw new Error("No NEXT_PUBLIC_SUPABASE_URL in env")
if (!supabaseAnonKey) throw new Error("No NEXT_PUBLIC_SUPABASE_KEY in env")

// Nur Supabase Client - OHNE LangChain!
export const supabase = createClient(supabaseUrl, supabaseAnonKey)


/**
 * sendet die relevantesten daten der session an ein backend. 
 * die daten sind: provider-token, refresh-token, userobjekt mit id und email (den erst nicht, nur diese 2 sind die wichtigsten)
 * WICHTIG: das ist eine async function, du muss sie awaiten!
 * @param toBackend = das ist die backend url an der du deine session schicken willst
 * @returns backend-response
 */
export async function sendSession<T>({toBackend,extraData}:{toBackend:string,extraData?:any}){
    const { data, error:sessionError } = await supabase.auth.getSession()
    console.log(`google_access_token:${data.session?.provider_token}`)
    console.log(`google_refresh_token:${data.session?.provider_refresh_token}`)
    console.log(`user_id und mail = id:${data.session?.user.id} + email:${data.session?.user.email}`)
    if (sessionError) throw new Error("fehler beim session Kriegen!")
    if (!data.session) {
        console.warn("Keine aktive Session - User ist nicht eingeloggt!")
        // Return a mock response or handle gracefully instead of throwing
        return {
            data: { message: "User not logged in", success: false } as T,
            status: 401
        }
    }
    try{
        const { data:backendData,status } = await axios.post<T>(
            toBackend,
            {
                google_access_token:data.session?.provider_token,
                google_refresh_token:data.session?.provider_refresh_token || null,
                user:{
                    id:data.session?.user.id,
                    email:data.session?.user.email
                },
                ...(extraData !== undefined && { extraData })
            }
        )
        return { data:backendData,status,session:data.session }
    }catch(e) {
        if (axios.isAxiosError(e)) {
            console.error(`Axios Fehler:, {
                status: ${e.response?.status},
                data: ${e.response?.data},
                message: ${e.message},
                code: ${e.code},
                config: {
                    url: ${e.config?.url},
                    method: ${e.config?.method},
                    baseURL: ${e.config?.baseURL}
                }
            }`)
            return { data: null, status: e.response?.status || 500 }
        } else {
            console.error("Unbekannter Fehler:", e)
            return { data: null, status: 500 }
        }
    }
}

export interface OAuthProps{
    provider?:Provider 
    scopes?:Array<string> | undefined
    redirectTo:string
}
/**
 * kümmert sich um den OAuth login prozess
 * WICHTIG: das ist eine async function, du muss sie awaiten!
 * @param provider - der OAuth provider den du nutzen möchtest, standardmäßig ist es "google"
 * @param scopes - die scopes die du für den OAuth login möchtest, als Array
 * @param redirectTo - die url auf die der user zurückgeleitet wird nach dem login, muss in supabase als autorisierte url eingetragen sein
 * @returns nichts, startet einfach den OAuth login prozess
 */
export async function OAuthLogin({provider = "google",scopes,redirectTo}:OAuthProps){
        const { error:SignInError } = await supabase.auth.signInWithOAuth({
            provider: provider,
            options: {
                scopes:scopes ? scopes.join(" ") : undefined,
                redirectTo:redirectTo
            }
        })
        if (SignInError) throw new Error("fehler beim OAuth Sign in!")  
    }

export async function addUser<T extends {user_id:string}>({toTable}:{toTable:SupabaseTable<T>}){
    const { data: { user }, error:getUserError } = await supabase.auth.getUser()
    if(getUserError) throw new Error("Error beim user kriegen in der 'addUser' function, Error: " + getUserError)
    if(!user) throw new Error("No user found")
    const user_id_obj = await toTable.select({
        columns:["user_id" as keyof T],
        where:[{column:"user_id" as keyof T, is:user.id}],
        first:true
    })
    if(user_id_obj) console.log("User mit der id: " + user.id + " ist bereits in der Tabelle: " + toTable.tableName + " vorhanden!")
    const user_obj = await toTable.insert([{user_id:user.id} as Partial<T>])
    if(!user_obj) throw new Error("Error beim user in die Tabelle: " + toTable.tableName + " hinzufügen, Error: " + user_obj)
    return true
}
