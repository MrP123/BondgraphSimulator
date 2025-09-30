import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import TreeLayout
import random
from uuid import uuid4

from pyBondGraph.bondgraph import (
    BondGraph,
    Bond,
    SourceEffort,
    Inductor,
    Capacitor,
    Resistor,
    Transformer,
    OneJunction,
    ZeroJunction,
    ElementOnePort,
    ElementTwoPort,
)

bond_graph = BondGraph()

voltage_source = SourceEffort("V", "V(t)")
capacitor = Capacitor("C6", "C6")
inductor = Inductor("I3", "L3")
resistor2 = Resistor("R2", "R2")
resistor7 = Resistor("R7", "R7")
transformer = Transformer("T1", "T1")
junction1 = OneJunction("J1")
junction0 = ZeroJunction("J0")

bond_graph.add_bond(Bond(voltage_source, junction1, "effort_out"))
bond_graph.add_bond(Bond(junction1, resistor2, "flow_out"))
bond_graph.add_bond(Bond(junction1, inductor, "effort_out"))
bond_graph.add_bond(Bond(junction1, transformer, "flow_out"))
bond_graph.add_bond(Bond(transformer, junction0, "flow_out"))
bond_graph.add_bond(Bond(junction0, capacitor, "flow_out"))
bond_graph.add_bond(Bond(junction0, resistor7, "effort_out"))

bond_graph.get_solution_equations()

st.set_page_config("Streamlit Flow Example", layout="wide")

st.title("Streamlit Flow Example")


if "curr_state" not in st.session_state:
    nodes = []
    for i, element in enumerate(bond_graph.elements):
        if isinstance(element, SourceEffort):
            nodes.append(StreamlitFlowNode(element.name, (i, 0), {'content': repr(element)}, 'input', source_position='right'))
        elif isinstance(element, ElementOnePort):
            nodes.append(StreamlitFlowNode(element.name, (i, 0), {'content': repr(element)}, 'output', target_position='left'))
        elif isinstance(element, ElementTwoPort):
            nodes.append(StreamlitFlowNode(element.name, (i, 0), {'content': repr(element)}, 'default', 'right', 'left'))
        else:
            nodes.append(StreamlitFlowNode(element.name, (i, 0), {'content': repr(element)}, 'default', 'right', 'left'))

    edges = []
    for bond in bond_graph.bonds:
        from_element, to_element = bond.elements
        edges.append(
            StreamlitFlowEdge(
                f"{from_element.name}-{to_element.name}",
                from_element.name,
                to_element.name,
                animated=False,
                marker_start={},
                marker_end={"type": "arrow", "color": "#000000", "strokeWidth": 1.5},
                label=f"{bond.num}",
                label_show_bg=True,
                label_bg_style={"fill": "#ffffff", "fillOpacity": 0.7},
                style={"stroke": "#000000", "strokeWidth": 1.5},
            )
        )

    st.session_state.curr_state = StreamlitFlowState(nodes, edges)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    element_type = st.selectbox("Element Type", [SourceEffort, Inductor, Capacitor, Resistor, Transformer, OneJunction, ZeroJunction], format_func=lambda x: x.__name__)
    with st.form("Add Element"):
        element_name = st.text_input("Element Name", value="")
        
        if element_type not in (OneJunction, ZeroJunction):
            element_value = st.text_input("Element Value (e.g. R, L, C, V(t)", value="")
        else:
            element_value = None

        submitted = st.form_submit_button("Add Element")

        if submitted and element_name:
            if element_type in (OneJunction, ZeroJunction):
                element = element_type(element_name)
            elif element_value:
                element = element_type(element_name, element_value)

            if isinstance(element, SourceEffort):
                st.session_state.curr_state.nodes.append(StreamlitFlowNode(element.name, (0, 0), {'content': repr(element)}, 'input', source_position='right'))
            elif isinstance(element, ElementOnePort):
                st.session_state.curr_state.nodes.append(StreamlitFlowNode(element.name, (0, 0), {'content': repr(element)}, 'output', target_position='left'))
            elif isinstance(element, ElementTwoPort):
                st.session_state.curr_state.nodes.append(StreamlitFlowNode(element.name, (0, 0), {'content': repr(element)}, 'default', 'right', 'left'))
            else:
                st.session_state.curr_state.nodes.append(StreamlitFlowNode(element.name, (0, 0), {'content': repr(element)}, 'default', 'right', 'left'))
            st.rerun()

    #if st.button("Add node"):
    #    new_node = StreamlitFlowNode(str(f"st-flow-node_{uuid4()}"), (0, 0), {'content': f'Node {len(st.session_state.curr_state.nodes) + 1}'}, 'default', 'right', 'left')
    #    st.session_state.curr_state.nodes.append(new_node)
    #    st.rerun()

with col2:
    if st.button("Delete Random Node"):
        if len(st.session_state.curr_state.nodes) > 0:
            node_to_delete = random.choice(st.session_state.curr_state.nodes)
            st.session_state.curr_state.nodes = [node for node in st.session_state.curr_state.nodes if node.id != node_to_delete.id]
            st.session_state.curr_state.edges = [edge for edge in st.session_state.curr_state.edges if edge.source != node_to_delete.id and edge.target != node_to_delete.id]
            st.rerun()

with col3:
    if st.button("Add Random Edge"):
        if len(st.session_state.curr_state.nodes) > 1:

            source_candidates = [streamlit_node for streamlit_node in st.session_state.curr_state.nodes if streamlit_node.type in ['input', 'default']]
            target_candidates = [streamlit_node for streamlit_node in st.session_state.curr_state.nodes if streamlit_node.type in ['default', 'output']]
            source = random.choice(source_candidates)
            target = random.choice(target_candidates)
            new_edge = StreamlitFlowEdge(f"{source.id}-{target.id}", source.id, target.id, animated=True)
            if not any(edge.id == new_edge.id for edge in st.session_state.curr_state.edges):
                st.session_state.curr_state.edges.append(new_edge)
            st.rerun()

with col4:
    if st.button("Delete Random Edge"):
        if len(st.session_state.curr_state.edges) > 0:
            edge_to_delete = random.choice(st.session_state.curr_state.edges)
            st.session_state.curr_state.edges = [edge for edge in st.session_state.curr_state.edges if edge.id != edge_to_delete.id]
            st.rerun()

with col5:
    if st.button("Random Flow"):
        nodes = [StreamlitFlowNode(str(f"st-flow-node_{uuid4()}"), (0, 0), {'content': f'Node {i}'}, 'default', 'right', 'left') for i in range(5)]
        edges = []
        for _ in range(5):
            source = random.choice(nodes)
            target = random.choice(nodes)
            if source.id != target.id:
                new_edge = StreamlitFlowEdge(f"{source.id}-{target.id}", source.id, target.id, animated=True)
                if not any(edge.id == new_edge.id for edge in edges):
                    edges.append(new_edge)
        st.session_state.curr_state = StreamlitFlowState(nodes=nodes, edges=edges)
        st.rerun()

st.session_state.curr_state = streamlit_flow(
    "example_flow",
    st.session_state.curr_state,
    layout=TreeLayout(direction="right"),
    fit_view=True,
    height=750,
    # enable_node_menu=True,
    # enable_edge_menu=True,
    # enable_pane_menu=True,
    # get_edge_on_click=True,
    # get_node_on_click=True,
    # show_minimap=True,
    show_controls=False,
    hide_watermark=True,
    allow_new_edges=True,
    min_zoom=0.1,
)


col1, col2, col3 = st.columns(3)

with col1:
    for node in st.session_state.curr_state.nodes:
        st.write(node)

with col2:
    for edge in st.session_state.curr_state.edges:
        st.write(edge)

with col3:
    st.write(st.session_state.curr_state.selected_id)
