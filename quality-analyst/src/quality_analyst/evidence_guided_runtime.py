"""Single guarded HTTP request, no retries, independent RAM/deadline monitor."""
import http.client,json,socket,threading,time
from quality_analyst import interactive
from quality_analyst.ollama import LocalOllama,INSTRUCT_PROFILE,AdapterError
from quality_analyst.release_config import settings,ollama_connection

class GuardedClient(LocalOllama):
    def __init__(self):
        super().__init__(profile=INSTRUCT_PROFILE)
        self.guard_samples=[];self.low_memory=False

    def guarded_post(self,payload):
        s=settings();limit=s.generation_timeout
        reserve=(1.5 if s.hardware_mode=='gpu' else s.cpu_ram_reserve_gib)*1024**3
        connection=ollama_connection(min(5,limit))
        stop=threading.Event();reason=[];active=[None];started=time.monotonic()
        def abort(code):
            if not reason:reason.append(code)
            self.low_memory=reason[0]=='RAM_GUARD_STOPPED_REQUEST'
            try:
                if active[0] is not None:active[0].shutdown(socket.SHUT_RDWR)
            except OSError:pass
            connection.close()
        def watch():
            while not stop.wait(.2):
                ram=interactive.hardware.probes.probe_ram();elapsed=time.monotonic()-started
                self.guard_samples.append({'elapsed':elapsed,'ram':ram})
                if ram.get('status')!='ok':abort('RAM_MONITOR_UNAVAILABLE');return
                if ram.get('available_bytes',0)<reserve:abort('RAM_GUARD_STOPPED_REQUEST');return
                if elapsed>=limit:abort('INFERENCE_TIMEOUT');return
        thread=threading.Thread(target=watch,daemon=True);thread.start()
        try:
            connection.connect();active[0]=connection.sock
            if reason:raise AdapterError(reason[0])
            active[0].settimeout(max(.001,limit-(time.monotonic()-started)))
            connection.request('POST','/api/generate',body=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode(),headers={'Content-Type':'application/json'})
            response=connection.getresponse();raw=response.read(131073)
            if reason:raise AdapterError(reason[0])
            if time.monotonic()-started>=limit:raise AdapterError('INFERENCE_TIMEOUT')
            if response.status!=200:
                self.http_error={'http_status':response.status,'body_omitted_for_privacy':True}
                if response.status in {400,413,500} and len(raw)<=131072:
                    from quality_analyst.contracts import strict_json_loads
                    try:
                        body=strict_json_loads(raw)
                        message=body.get('error','') if isinstance(body,dict) else ''
                        if isinstance(message,str) and 'context' in message.lower() and any(
                            word in message.lower() for word in ('exceed','too long','length')):
                            raise AdapterError('CONTEXT_LIMIT_NO_TRUNCATION')
                        # Ollama could not load the model weights: host RAM, not the request.
                        if isinstance(message,str) and 'failed to allocate' in message.lower():
                            raise AdapterError('OLLAMA_OUT_OF_MEMORY',http_status=response.status)
                    except (ValueError,RecursionError):pass
                raise AdapterError('HTTP_ERROR',http_status=response.status)
            if len(raw)>131072:raise AdapterError('RESPONSE_TOO_LARGE')
            from quality_analyst.contracts import strict_json_loads
            try:envelope=strict_json_loads(raw)
            except (ValueError,RecursionError):raise AdapterError('INVALID_ENVELOPE_JSON') from None
            if not isinstance(envelope,dict):raise AdapterError('INVALID_ENVELOPE_JSON')
            envelope.pop('thinking',None);return envelope
        except AdapterError:raise
        except (TimeoutError,socket.timeout):raise AdapterError(reason[0] if reason else 'INFERENCE_TIMEOUT') from None
        except (OSError,http.client.HTTPException):
            raise AdapterError(reason[0] if reason else 'SERVER_UNAVAILABLE') from None
        finally:
            stop.set();thread.join(timeout=2);connection.close()
