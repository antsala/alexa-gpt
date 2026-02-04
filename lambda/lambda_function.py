from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response
import ask_sdk_core.utils as ask_utils
import requests
import logging
import json
import re

# Clave de API de OpenAI
api_key = "YOUR_API_KEY"

model = "gpt-4o-mini"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LaunchRequestHandler(AbstractRequestHandler):
    """Handler para el lanzamiento de la skill."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Modo Chat G.P.T. activado."

        session_attr = handler_input.attributes_manager.session_attributes
        session_attr["chat_history"] = []

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )

class GptQueryIntentHandler(AbstractRequestHandler):
    """Handler para la intención GptQueryIntent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("GptQueryIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        query = handler_input.request_envelope.request.intent.slots["query"].value

        session_attr = handler_input.attributes_manager.session_attributes
        if "chat_history" not in session_attr:
            session_attr["chat_history"] = []
            session_attr["last_context"] = None
        
        # Procesar la consulta para detectar si es de seguimiento
        processed_query, is_followup = process_followup_question(query, session_attr.get("last_context"))
        
        # Generar respuesta con manejo de contexto
        response_data = generate_gpt_response(session_attr["chat_history"], processed_query, is_followup)
        
        # La función puede devolver una tupla (texto, preguntas) o solo texto
        if isinstance(response_data, tuple) and len(response_data) == 2:
            response_text, followup_questions = response_data
        else:
            # Fallback ante errores
            response_text = str(response_data)
            followup_questions = []
        
        # Guardar preguntas de seguimiento en sesión
        session_attr["followup_questions"] = followup_questions
        
        # Actualizar historial (solo texto de respuesta, sin preguntas)
        session_attr["chat_history"].append((query, response_text))
        session_attr["last_context"] = extract_context(query, response_text)
        
        # Formatear la respuesta con sugerencias si las hay
        response = response_text
        if followup_questions and len(followup_questions) > 0:
            # Pausa breve antes de las sugerencias
            response += " <break time=\"0.5s\"/> "
            response += "Podrías preguntar: "
            # Unir con 'o' la última pregunta
            if len(followup_questions) > 1:
                response += ", ".join([f"'{q}'" for q in followup_questions[:-1]])
                response += f", o '{followup_questions[-1]}'"
            else:
                response += f"'{followup_questions[0]}'"
            response += ". <break time=\"0.5s\"/> ¿Qué te gustaría saber?"
        
        # Reprompt (con y sin sugerencias)
        reprompt_text = "Puedes hacerme otra pregunta o decir 'para' para terminar la conversación."
        if 'followup_questions' in session_attr and session_attr['followup_questions']:
            reprompt_text = "Puedes hacerme otra pregunta, decir 'siguiente' para escuchar más sugerencias o decir 'para' para terminar la conversación."
        
        return (
            handler_input.response_builder
                .speak(response)
                .ask(reprompt_text)
                .response
        )

class CatchAllExceptionHandler(AbstractExceptionHandler):
    """Manejo genérico de errores de sintaxis o enrutado."""
    def can_handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> bool
        return True

    def handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> Response
        logger.error(exception, exc_info=True)

        speak_output = "Lo siento, he tenido problemas para hacer lo que pediste. Inténtalo de nuevo."

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )

class CancelOrStopIntentHandler(AbstractRequestHandler):
    """Handler único para Cancel y Stop."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return (ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input) or
                ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Saliendo del modo Chat G.P.T."

        return (
            handler_input.response_builder
                .speak(speak_output)
                .response
        )

def process_followup_question(question, last_context):
    """Determina si la pregunta es de seguimiento y añade contexto si es necesario."""
    # Indicadores comunes de seguimiento (en inglés por compatibilidad con consultas mixtas)
    followup_patterns = [
        r'^(what|how|why|when|where|who|which)\s+(about|is|are|was|were|do|does|did|can|could|would|should|will)\s',
        r'^(and|but|so|then|also)\s',
        r'^(can|could|would|should|will)\s+(you|it|they|we)\s',
        r'^(is|are|was|were|do|does|did)\s+(it|that|this|they|those|these)\s',
        r'^(tell me more|elaborate|explain further)\s*',
        r'^(why|how)\?*$'
    ]
    
    is_followup = False
    
    for pattern in followup_patterns:
        if re.search(pattern, question.lower()):
            is_followup = True
            break
    
    return question, is_followup

def extract_context(question, response):
    """Extrae un contexto básico de una pareja P&R para referencia futura."""
    return {"question": question, "response": response}

def generate_followup_questions(conversation_context, query, response, count=2):
    """Genera preguntas de seguimiento concisas según el contexto de conversación."""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        url = "https://api.openai.com/v1/chat/completions"
        
        # Prompt enfocado a preguntas breves de seguimiento
        messages = [
            {"role": "system", "content": "Eres un asistente útil que sugiere preguntas de seguimiento cortas."},
            {"role": "user", "content": """Según la conversación, sugiere 2 preguntas de seguimiento muy breves (máx. 4 palabras cada una).
Hazlas directas y simples. Devuelve SOLO las preguntas separadas por '|'.
Ejemplo: ¿Cuál es la capital?|¿Qué tamaño tiene?"""}
        ]
        
        # Añadir contexto de conversación
        if conversation_context:
            last_q, last_a = conversation_context[-1]
            messages.append({"role": "user", "content": f"Pregunta anterior: {last_q}"})
            messages.append({"role": "assistant", "content": last_a})
        
        messages.append({"role": "user", "content": f"Pregunta actual: {query}"})
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": "Preguntas de seguimiento (separadas por |):"})
        
        data = {
            "model": "gpt-3.5-turbo",  # Modelo rápido para las sugerencias
            "messages": messages,
            "max_completion_tokens": 50,
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=3)
        if response.ok:
            questions_text = response.json()['choices'][0]['message']['content'].strip()
            # Limpiar y dividir
            questions = [q.strip().rstrip('?') for q in questions_text.split('|') if q.strip()]
            # Validar longitud
            questions = [q for q in questions if len(q.split()) <= 4 and len(q) > 0][:2]
            
            # Si no hay suficientes, usar por defecto (en español)
            if len(questions) < 2:
                questions = ["Dime más", "Pon un ejemplo"]
                
            logger.info(f"Generated follow-up questions: {questions}")
            return questions
            
        logger.error(f"API Error: {response.text}")
        return ["Dime más", "Pon un ejemplo"]
        
    except Exception as e:
        logger.error(f"Error in generate_followup_questions: {str(e)}")
        return ["Dime más", "Pon un ejemplo"]

def generate_gpt_response(chat_history, new_question, is_followup=False):
    """Genera una respuesta GPT con manejo de contexto mejorado."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    url = "https://api.openai.com/v1/chat/completions"
    
    # Mensaje del sistema, ajustado si es seguimiento
    system_message = "Eres un asistente útil. Responde en 50 palabras o menos."
    if is_followup:
        system_message += " Esta es una pregunta de seguimiento. Mantén el contexto sin repetir información ya dada."
    
    messages = [{"role": "system", "content": system_message}]
    
    # Incluir historial (menos cuando no es seguimiento, para ahorrar tokens)
    history_limit = 10 if not is_followup else 5
    for question, answer in chat_history[-history_limit:]:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    
    # Añadir la nueva pregunta
    messages.append({"role": "user", "content": new_question})
    
    data = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": 300
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response_data = response.json()
        if response.ok:
            response_text = response_data['choices'][0]['message']['content']
            
            # Generar preguntas de seguimiento siempre
            try:
                followup_questions = generate_followup_questions(
                    chat_history + [(new_question, response_text)], 
                    new_question, 
                    response_text
                )
                logger.info(f"Generated follow-up questions: {followup_questions}")
            except Exception as e:
                logger.error(f"Error generating follow-up questions: {str(e)}")
                followup_questions = []
            
            return response_text, followup_questions
        else:
            return f"Error {response.status_code}: {response_data['error']['message']}", []
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error al generar la respuesta: {str(e)}", []

class ClearContextIntentHandler(AbstractRequestHandler):
    """Handler para limpiar el contexto de la conversación."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("ClearContextIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        session_attr = handler_input.attributes_manager.session_attributes
        session_attr["chat_history"] = []
        session_attr["last_context"] = None
        
        speak_output = "He borrado nuestro historial de conversación. ¿Sobre qué te gustaría hablar?"
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )

sb = SkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(GptQueryIntentHandler())
sb.add_request_handler(ClearContextIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
