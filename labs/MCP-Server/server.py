import json
from typing_extensions import Any
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="Course Assistant",
    instructions="""
        Return course information using mcp tools
    """
)

DATA_FILE = Path(__file__).resolve().parent/"data"/"courses.json"

def _loadCourses()->list[dict[str,Any]]:
    """Return all course information in a list"""
    try:
        with DATA_FILE.open(encoding="utf-8") as file:
            courses = json.load(file)
            return courses
    except Exception as e:
        raise RuntimeError(f"File opening is failed. {e}")

def _normalizedCourseID(courseID:str)->str:
    """Normalize the course ID"""
    normalizedCourseID = "".join(courseID.upper().split())
    return normalizedCourseID

@mcp.tool()
def get_course_list()->list[dict[str,Any]]:
    """Return list of available courses"""
    return[
        {
            "course code":course["course_id"],
            "title":course["title"],
            "credit hours": course["credit_hours"]
        }
        for course in _loadCourses()
    ]

@mcp.tool()
def get_course_detail(courseID:str)->dict[str,Any]:
    """Return details of a given course"""
    for course in _loadCourses():
        if _normalizedCourseID(course["course_id"]) == _normalizedCourseID(courseID):
            return course

    return{
        "Error":f"The {courseID} is not found",
        "Available courses":[course["course_id"] for course in _loadCourses()]
    }
if __name__ == "__main__":
    mcp.run(transport="stdio")