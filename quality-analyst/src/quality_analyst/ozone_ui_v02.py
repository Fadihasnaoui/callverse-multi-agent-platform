"""Conversation -> actual public Ozone search -> free qualitative diagnostic."""
import json,threading
from quality_analyst.release_config import settings
from datetime import datetime,timezone
import streamlit as st
from quality_analyst import ozone_service_v02 as service
from quality_analyst.web_domain import import_bytes,rebuild
from quality_analyst.interactive import stable_conversation,InputError
from quality_analyst.ollama import compact,sha256
from quality_analyst.ozone_live.search import search_hotels,make_spec,empty
from quality_analyst.ozone_live.real_contracts import SearchResult


def fingerprint(c):return sha256(compact(c.model_dump(mode='json')).encode())

def install(c):
    for key in list(st.session_state):
        if key.startswith('x_'):st.session_state.pop(key,None)
    st.session_state.update(x_original=c,x_rows=[{'role':m.role.value,'text':m.text} for m in c.messages],
                            x_language=c.language.value,x_complete=c.conversation_complete)

def render_source(result):
    st.caption('État : '+result.data_status+' · origine : '+result.origin+' · '+f'{result.duration_seconds:.2f} s HTTP')
    for error in result.errors:st.warning(error)
    if result.fetched_at:
        age=(datetime.now(timezone.utc)-result.fetched_at).total_seconds()
        st.caption('Récupération : '+result.fetched_at.isoformat()+f' · âge {age:.0f} s · cache : {result.cache_hit}')
        if age>settings().ozone_cache_ttl:st.warning('Capture datée : disponibilité périmée pour une nouvelle vérification courante.')
    if result.offers:
        st.dataframe([{'Hôtel':o.hotel_name,'Étoiles':o.stars,
            'Prix affiché':(str(o.amount)+' '+(o.currency or '')) if o.amount is not None else None,
            'Base':o.price_basis,'Nuits':o.nights,'Disponibilité':o.availability,'Offres affichées':o.offer_count,
            'Chambre':o.room_type,'Formule':o.board,'Détails':o.detail_url} for o in result.offers],hide_index=True,
            column_config={'Détails':st.column_config.LinkColumn('Détails')})
    else:st.info('Aucun hôtel observé ; cela ne confirme aucune indisponibilité.')
    st.caption(f'{len(result.offers)} fiches extraites (maximum 12) ; compteur déclaré par la source : {result.reported_results}. Pas de pagination.')
    st.caption('Prix de synthèse hôtel : aucune chambre, formule ou offre fournisseur sélectionnée ; aucune réservation garantie.')
    with st.expander('Provenance, paramètres et preuves Ozone'):
        st.json(result.model_dump(mode='json'))

def render_provider_error(error):
    messages={
        'GEMINI_KEY_NOT_CONFIGURED':'Clé Gemini absente. Configurez le fichier secret puis recréez le service applicatif.',
        'GEMINI_KEY_FILE_INVALID':'Le fichier secret Gemini est invalide.',
        'GEMINI_KEY_FILE_UNREADABLE':'Le fichier secret Gemini ne peut pas être lu par l’application.',
        'GEMINI_QUOTA_EXHAUSTED_NO_RETRY':'Quota Gemini épuisé ou débit limité (429). Aucun retry ni changement de moteur. Consultez vos limites avant un nouvel essai manuel.',
        'GEMINI_LOCAL_DAILY_LIMIT_REACHED':'La limite quotidienne locale d’appels Gemini est atteinte. Aucun appel envoyé.',
        'GEMINI_AUTH_OR_ACCESS_REFUSED':'Google refuse la clé ou les droits du projet. Aucun autre compte ou moteur utilisé.',
        'GEMINI_MODEL_UNAVAILABLE':'Le modèle Gemini configuré n’est pas disponible pour ce compte. Aucun modèle substitué.',
        'GEMINI_TIMEOUT_NO_RETRY':'Délai Gemini dépassé. Aucun retry ; vérifier la session et le verrou avant toute intervention.',
        'GEMINI_REQUEST_OR_SCHEMA_REFUSED':'Google a refusé la requête ou son schéma. Aucun retry ni réparation automatique.',
        'GEMINI_RESPONSE_INCOMPLETE_OR_BLOCKED':'Réponse Gemini incomplète ou bloquée ; aucun rapport accepté.',
        'GEMINI_CONSENT_REQUIRED':'Votre accord explicite est requis avant l’envoi à Google.',
        'GEMINI_UNBILLED_PROJECT_CONFIRMATION_REQUIRED':'Confirmez l’utilisation d’un projet Google sans facturation activée.',
    }
    st.error(messages.get(error,error))

def cloud_agreement(key):
    st.info('Gemini transmet à Google la conversation, les critères et les preuves normalisées nécessaires à cette étape. Ce traitement est effectué dans le cloud. Selon les conditions applicables, Google peut utiliser les entrées et sorties du service gratuit pour améliorer ses produits et les faire examiner par des personnes. Retirez les données sensibles et consultez les [conditions Google pour votre région](https://ai.google.dev/gemini-api/terms).')
    return st.checkbox('J’autorise cet envoi à Google et je confirme que le projet utilisé n’a pas de facturation activée.',key=key)

def render_report(result):
    st.caption(result.get('provider','ollama')+' · '+result['engine']+' · '+result['stage']+f" · {result.get('wall_seconds',0):.2f} s · {result.get('generation_calls')} appel(s)")
    for error in result.get('errors',[]):render_provider_error(error)
    checked=result.get('checked') or {};proposal=checked.get('proposal');v=checked.get('verification') or {}
    # Inspection of immutable saved outputs, never a second inference or a
    # repaired proposal. Source checks remain visible when model schema fails.
    directory=(service.SESSIONS/result.get('session_id','')).resolve()
    comparisons=v.get('factual_checks',[]);packet={}
    if directory.is_relative_to(service.SESSIONS.resolve()) and (directory/'input.json').is_file():
        packet=json.loads((directory/'input.json').read_text(encoding='utf-8'))
        path=directory/'runtime/comparisons.json'
        if not comparisons and path.is_file():comparisons=json.loads(path.read_text(encoding='utf-8'))
        raw=directory/'runtime/raw_response.txt'
        if raw.is_file():
            with st.expander('Réponse IA brute — non validée sémantiquement'):
                st.text(raw.read_text(encoding='utf-8'))
    if comparisons:
        with st.expander('Comparaisons calculées et preuves Ozone datées',expanded=True):
            by_id={e['evidence_id']:e for e in (packet.get('retrieval') or {}).get('evidence',[])}
            for comparison in comparisons:
                st.write(comparison['comparison_id']+' · '+comparison['status'])
                st.caption(comparison['reason']+' · '+str(comparison['fetched_at']))
                for key in comparison['evidence_ids']:
                    if key in by_id:
                        e=by_id[key];st.caption(e['field']+' · '+e['source_url']+' · '+e['fetched_at']);st.text(e['observed'])
                st.caption('Comparaison d’un affichage daté ; aucune faute historique ou réservation prouvée.')
    if not proposal:
        st.warning('Réponse structurée inexploitable ; aucun retry ni résultat substitué.')
        if checked.get('reason'):st.error(checked['reason'])
        return
    st.write('Besoin du client');st.write(proposal['needs'])
    st.write('Traitement par le conseiller');st.write(proposal['handling'])
    rows=v.get('weaknesses',[])
    if not rows:st.info('Aucune faiblesse proposée. Ce résultat ne certifie pas une absence de défaut.')
    for row in rows:
        w=row['weakness']
        with st.expander(w['title'],expanded=row['status']!='rejected'):
            st.caption('Hypothèse à revoir — aucune erreur du conseiller n’est confirmée automatiquement.')
            st.write(w['description']);st.write('Conséquence : '+w['consequence']);st.write('Amélioration : '+w['recommendation'])
            for q in w['citations']:
                st.caption(q['message_id']+' · '+q['role']);st.text(q['quote'])
            for issue in row['issues']:st.error('Proposition rejetée : '+issue)
            st.caption('Comparaisons Ozone : '+(', '.join(w['comparison_ids']) or 'aucune'))
    st.write('Vérifications factuelles datées')
    for comparison in v.get('factual_checks',[]):
        st.write(comparison['comparison_id']+' · '+comparison['status'])
        st.caption(comparison['reason']+' · '+str(comparison['fetched_at']))
        st.caption('Identité : '+str(comparison.get('identity_status'))+' · aucune preuve d’erreur historique.')
    st.write('Points non vérifiables')
    for limit in proposal.get('unverifiable',[]):st.write(limit)
    with st.expander('Audit exact des citations'):
        st.json(v.get('checks',[]))

def main():
    st.set_page_config(page_title='Quality Analyst — Ozone Live v0.2',layout='wide')
    # Presentation-only shared Ozone Digital Twin design tokens. Diagnostic
    # rules, provider controls and evidence rendering remain unchanged.
    st.markdown('''<style>
      :root{--ozone-ink:#16323b;--ozone-violet:#6859de;}
      .stApp{background:radial-gradient(circle at 92% 0,#e7e4ff 0,transparent 26%),#f6f9f8;color:#173039;}
      [data-testid="stSidebar"]{background:linear-gradient(180deg,#16323b,#233664);border-right:1px solid #36535a;}
      [data-testid="stSidebar"] *{color:#e7f1ef!important;}
      [data-testid="stSidebar"] .stSelectbox div{background:#243e49!important;border-color:#45616a!important;}
      h1,h2,h3{color:#16323b!important;letter-spacing:-.025em;}
      .stButton>button{border:0!important;border-radius:10px!important;background:linear-gradient(110deg,#285d61,#6859de)!important;color:#fff!important;font-weight:700!important;box-shadow:0 8px 18px #3a347522!important;}
      .stTextInput input,.stTextArea textarea,.stSelectbox div{border-radius:10px!important;border-color:#cadada!important;background:#fff!important;}
      [data-testid="stExpander"],.stDataFrame{border:1px solid #dbe6e6!important;border-radius:14px!important;background:#fff!important;box-shadow:0 10px 24px #1730390c!important;}
      [data-testid="stAlert"]{border-radius:12px!important;}
      @media (prefers-color-scheme:dark){.stApp{background:#101b20;color:#eaf4f1;}h1,h2,h3{color:#edf7f3!important;}[data-testid="stExpander"],.stDataFrame{background:#17262c!important;border-color:#30494d!important;}.stTextInput input,.stTextArea textarea,.stSelectbox div{background:#17262c!important;color:#edf7f3!important;}}
    </style>''', unsafe_allow_html=True)
    st.sidebar.caption('Ozone Live v0.2 · distribution · diagnostic_only')
    st.title('Quality Analyst — Ozone Live v0.2')
    st.warning('Analyse IA diagnostique : citations contrôlées, interprétations non validées indépendamment. Aucun jugement autonome admis.')
    labels={'gemini':'Gemini — Cloud, recommandé pour les machines modestes.',
        'ollama':'Qwen — Local, nécessite Ollama et des ressources suffisantes.'}
    provider=st.sidebar.selectbox('Moteur d’intelligence artificielle',list(labels),
        index=list(labels).index(settings().ai_provider),format_func=lambda p:labels[p],key='ai_provider')
    model=settings().gemini_model if provider=='gemini' else settings().ollama_model
    provider_key=sha256(compact({'provider':provider,'model':model}).encode())[:16]
    st.sidebar.caption('Fournisseur actif : '+('Gemini / Google Cloud' if provider=='gemini' else 'Qwen / Ollama local'))
    st.sidebar.caption('Modèle configuré : '+model)
    if provider=='gemini':st.sidebar.caption('Quota gratuit limité selon le compte et la région. La facturation du projet n’est pas vérifiable par cette application ; aucun mécanisme d’activation ou de repli payant n’est utilisé.')
    st.caption('Recherches publiques ponctuelles à votre demande ; aucun balayage, réservation ou export d’entraînement. Aucun export automatique vers un corpus.')
    if 'x_rows' not in st.session_state:
        st.session_state.update(x_rows=[{'role':'customer','text':''},{'role':'advisor','text':''}],x_original=None)
    st.header('1. Conversation')
    st.caption('Retirer secrets et données confidentielles avant saisie. L’original ne sera pas réécrit par les critères de recherche.')
    upload=st.file_uploader('Importer une conversation JSON',type=['json'],key='x_upload')
    if st.button('Charger le JSON',key='x_import',disabled=upload is None):
        try:install(import_bytes(upload.getvalue()));st.rerun()
        except (InputError,ValueError):st.error('Import refusé ; aucun message supprimé ou corrigé.')
    st.selectbox('Langue',['fr','en'],key='x_language')
    if 'x_complete' not in st.session_state:st.session_state['x_complete']=True
    st.checkbox('Capture complète',key='x_complete')
    rows=[]
    roles=['customer','advisor','tool','system','other']
    for i,row in enumerate(st.session_state['x_rows']):
        if 'x_role_'+str(i) not in st.session_state:st.session_state['x_role_'+str(i)]=row['role']
        if 'x_text_'+str(i) not in st.session_state:st.session_state['x_text_'+str(i)]=row['text']
        role=st.selectbox('Rôle '+str(i+1),roles,key='x_role_'+str(i))
        text=st.text_area('Message '+str(i+1),key='x_text_'+str(i))
        rows.append({'role':role,'text':text})
    st.session_state['x_rows']=rows
    if st.button('Ajouter un message',key='x_add'):
        st.session_state['x_rows'].append({'role':'advisor','text':''});st.rerun()
    c=None
    try:
        if all(r['text'].strip() for r in rows):
            original=st.session_state.get('x_original')
            if original and [{'role':m.role.value,'text':m.text} for m in original.messages]==rows and original.language.value==st.session_state['x_language'] and original.conversation_complete==st.session_state['x_complete']:
                c=original
            else:c=stable_conversation(rows,st.session_state['x_language'],st.session_state['x_complete'])
    except (ValueError,InputError):st.error('Conversation invalide ; aucun appel.')
    source_hash=fingerprint(c) if c else None
    context_consent=cloud_agreement('x_cloud_context_'+provider_key+'_'+str(source_hash)) if provider=='gemini' and c else False
    if st.button('Détecter les critères de voyage',key='x_detect',disabled=c is None or (provider=='gemini' and not context_consent)):
        with st.spinner('Extraction '+('Gemini cloud' if provider=='gemini' else 'Qwen locale')+', sans retry…'):r=service.run(c,stage='context',provider=provider,cloud_consent=context_consent,free_tier_confirmed=context_consent)
        st.session_state['x_context']={'source_hash':source_hash,'provider_key':provider_key,'result':r}
        st.session_state.pop('x_analysis',None)
        verified=((r.get('checked') or {}).get('verification') or {})
        for row in verified.get('fields',[]):
            f=row['field']
            if row['accepted'] and f['owner']=='customer_request' and f['field'] in {'destination','check_in','check_out','hotel','room_type','board','services','budget'}:
                key='x_search_'+f['field']
                if not st.session_state.get(key):st.session_state[key]=f['value']
    context_result=st.session_state.get('x_context')
    context=None
    if context_result and context_result['source_hash']==source_hash and context_result.get('provider_key')==provider_key:
        context=((context_result['result'].get('checked') or {}).get('verification'))
        if context:
            st.dataframe([{'Champ':r['field']['field'],'Valeur':r['field']['value'],'Origine':r['field']['owner'],
                'Citation':r['field']['citation']['quote'],'Recevable techniquement':r['accepted']} for r in context['fields']],hide_index=True)
            for missing in context.get('missing_information',[]):st.caption('À préciser : '+missing)
        else:
            st.warning('Extraction inexploitable : paramètres manuels et analyse conversationnelle restent possibles.')
            for error in context_result['result'].get('errors',[]):
                if error in {'CONTEXT_LIMIT_NO_TRUNCATION','TURN_LIMIT_NO_TRUNCATION'}:
                    st.error('Conversation trop longue pour ce profil ; aucun texte tronqué et aucun appel IA lancé.')
                else:render_provider_error(error)
    st.header('2. Ozone Live')
    detected={}
    for row in (context or {}).get('fields',[]):
        f=row['field']
        if row['accepted'] and f['owner']=='customer_request':detected.setdefault(f['field'],f['value'])
    values={}
    for key,label in [('destination','Destination'),('check_in','Arrivée ISO AAAA-MM-JJ'),('check_out','Départ ISO AAAA-MM-JJ'),
        ('hotel','Hôtel souhaité'),('room_type','Chambre souhaitée'),('board','Formule souhaitée'),('services','Services demandés'),('budget','Budget')]:
        state_key='x_search_'+key
        if state_key not in st.session_state:st.session_state[state_key]=detected.get(key,'')
        values[key]=st.text_input(label,key=state_key)
    occupancy=st.text_area('Chambres : adultes et âges de chaque enfant (JSON)',placeholder='[{"adults": 2, "child_ages": []}]',key='x_occupancy')
    st.caption('Aucune occupation par défaut. Destination, dates et occupation sont envoyées à Ozone. Hôtel souhaité, type de chambre, formule, services et budget servent à l’analyse ; ils ne filtrent pas cette recherche de synthèses par destination.')
    form_hash=sha256(compact({'values':values,'rooms':occupancy}).encode())
    if st.button('Valider mes paramètres',key='x_validate',disabled=c is None):
        try:
            spec=make_spec(values['destination'],values['check_in'],values['check_out'],json.loads(occupancy))
            st.session_state['x_spec']={'source_hash':source_hash,'form_hash':form_hash,'spec':spec.model_dump(mode='json')}
            st.success('Critères confirmés, sans modifier la conversation.')
        except (ValueError,TypeError,KeyError):st.error('Destination, dates ou occupation manquantes/incohérentes. Aucun paramètre inventé.')
    saved=st.session_state.get('x_spec');spec=None
    if saved and saved['source_hash']==source_hash and saved['form_hash']==form_hash:
        from quality_analyst.ozone_live.contracts import SearchSpec
        spec=SearchSpec.model_validate_json(compact(saved['spec']))
    if st.button('Annuler la recherche en cours',key='x_cancel',disabled=st.session_state.get('x_cancel_event') is None):
        event=st.session_state.get('x_cancel_event')
        if event:event.set()
    if st.button('Rechercher les hôtels sur Ozone',key='x_fetch',disabled=spec is None):
        event=threading.Event();st.session_state['x_cancel_event']=event
        try:
            with st.spinner('Recherche HTTP publique bornée, sans réservation…'):
                r=search_hotels(spec.destination,spec.check_in,spec.check_out,[room.model_dump() for room in spec.rooms],cancel=event)
            st.session_state['x_retrieval']={'source_hash':source_hash,'form_hash':form_hash,'result':r.model_dump(mode='json')}
            st.session_state.pop('x_analysis',None)
        finally:st.session_state['x_cancel_event']=None
    retrieval=empty(spec,'NO_SEARCH_EXECUTED')
    saved=st.session_state.get('x_retrieval')
    if saved and saved['source_hash']==source_hash and saved['form_hash']==form_hash:
        retrieval=SearchResult.model_validate_json(compact(saved['result']))
    render_source(retrieval)
    chain_key=sha256(compact({'provider':provider_key,'source':source_hash,'form':form_hash,'retrieval':retrieval.search_session_id,
        'context':context_result['result']['session_id'] if context_result and context_result['source_hash']==source_hash and context_result.get('provider_key')==provider_key else None}).encode())
    st.header('3. Rapport Quality Analyst')
    analysis_consent=cloud_agreement('x_cloud_analysis_'+chain_key) if provider=='gemini' and c else False
    if st.button('Analyser la qualité du conseiller',key='x_analyze',disabled=c is None or (provider=='gemini' and not analysis_consent)):
        analysis_context={**(context or {}),'user_criteria':{**values,'rooms':occupancy,
            'provenance':'manual_user_confirmation_not_transcript_text',
            'room_board_hotel_services_budget_not_applied_as_source_filters':True}}
        with st.spinner('Analyse qualitative libre, un appel '+('Google cloud' if provider=='gemini' else 'local')+'…'):r=service.run(c,stage='analysis',context=analysis_context,retrieval=retrieval,provider=provider,cloud_consent=analysis_consent,free_tier_confirmed=analysis_consent)
        st.session_state['x_analysis']={'chain_key':chain_key,'result':r}
    analysis=st.session_state.get('x_analysis')
    if analysis and analysis['chain_key']==chain_key:render_report(analysis['result'])
    else:st.caption('Sans recherche utilisable, seule la conversation pourra être examinée ; les faits externes resteront non vérifiables.')
    history=service.history()
    with st.expander('Historique et comparaison des sessions'):
        st.dataframe([{'Session':r['session_id'],'Fournisseur':r.get('provider','ollama'),'Étape':r['stage'],'Statut':r['status'],'Appels':r.get('generation_calls'),
            'Durée':r.get('wall_seconds')} for r in history],hide_index=True)
        reports={r['session_id']:r for r in history if r['stage']=='analysis'}
        options=['']+list(reports)
        left=st.selectbox('Rapport A',options,key='x_compare_a');right=st.selectbox('Rapport B',options,key='x_compare_b')
        if left and right:
            if reports[left]['input_sha256']!=reports[right]['input_sha256']:st.warning('Conversations différentes : comparaison descriptive, sans cible commune.')
            for column,key in zip(st.columns(2),[left,right]):
                with column:render_report(reports[key])
