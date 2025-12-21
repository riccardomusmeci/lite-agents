import inspect
import textwrap
from typing import (
    Callable, 
    Any, 
    Dict, 
    List, 
    Union, 
    get_origin, 
    get_args
)

class Tool:
    """A decorator that wraps a function and adds additional functionality to it,
    making it suitable for use as an agent tool.
    
    Args:
        function (Callable): The function to be decorated.
    """
    def __init__(self, function: Callable):
        self.tool = function
        self._signature = inspect.signature(self.tool)

    @property
    def description(self) -> str:
        """Return the tool's source code definition.
        
        Returns:
            str: The tool's source code with the @Tool decorator removed.
        """
        try:
            code = inspect.getsource(self.tool)
            # Remove the decorator usage to avoid confusion in the prompt
            code = code.replace("@Tool", "")
            return textwrap.dedent(code).strip()
        except OSError:
            return "Source code not available."
    
    @property
    def name(self) -> str:
        """Return the tool's name.
        
        Returns:
            str: The function name.
        """
        return self.tool.__name__
    
    @property
    def docstring(self) -> str:
        """Return the tool's docstring.
        
        Returns:
            str: The function docstring or a default message.
        """
        return inspect.getdoc(self.tool) or "No docstring available."
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        """Generate the JSON schema for the tool's input arguments.
        
        Returns:
            Dict[str, Any]: A JSON schema dictionary.
        """
        properties = {}
        required = []
        
        for param_name, param in self._signature.parameters.items():
            if param_name == 'self':
                continue
            
            properties[param_name] = self._get_type_schema(param.annotation)
            
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
                
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    @staticmethod
    def _get_type_schema(py_type: Any) -> Dict[str, Any]:
        """Map Python types to JSON schema types.
        
        Args:
            py_type (Any): The Python type annotation.
            
        Returns:
            Dict[str, Any]: The corresponding JSON schema fragment.
        """
        origin = get_origin(py_type)
        args = get_args(py_type)
        
        # Handle Union types (e.g., Optional[int], Union[str, int])
        if origin is Union or str(origin) == "<class 'types.UnionType'>": # Handle | operator in Python 3.10+
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                return Tool._get_type_schema(non_none_args[0])
            # For multiple types, we default to string or could use "anyOf"
            return {"type": "string"}
        
        # Handle Lists
        if origin in (list, List) or py_type is list:
            schema = {"type": "array"}
            if args:
                schema["items"] = Tool._get_type_schema(args[0])
            return schema
            
        # Handle Dicts
        if origin in (dict, Dict) or py_type is dict:
            return {"type": "object"}
        
        # Handle primitive types
        type_mapping = {
            int: "integer",
            float: "number",
            bool: "boolean",
            str: "string"
        }
        
        if py_type in type_mapping:
            return {"type": type_mapping[py_type]}
            
        # Fallback
        return {"type": "string"}

    def execute(self, *args, **kwargs) -> Any:
        """Execute the decorated function.
        
        Returns:
            Any: The result of the function execution or the error message.
        """
        try:
            return self.tool(*args, **kwargs)
        except Exception as e:
            return f"Error executing tool {self.name}: {str(e)}"

    def __call__(self, *args, **kwargs) -> Any:
        """Make the decorator itself callable to maintain the expected behavior of a decorator.
        
        Returns:
            Any: The result of the tool's execution.
        """
        return self.execute(*args, **kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the tool metadata.
        
        Returns:
            Dict[str, Any]: Tool metadata including name, description, and schema.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.docstring,
                "parameters": self.input_schema
            }
        }