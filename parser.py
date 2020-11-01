# -*- coding: utf-8 -*-
"""parse_data.ipynb

Automatically generated by Colaboratory.

"""

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/My Drive/Colab Notebooks/GraphNN'

AND_nodes = ["Sitting Furniture", "Chair Head", "Chair Back",
             "Chair Arm", "Chair Seat", "Regular Leg Base",
             "Star-shape Leg Base", "Mechanical Control", "Footrest Ring",
             "Pedestal Base", "Foot Base", "Surface Base", "Star-shape Leg Set",
             "Wheel Assembly"]
OR_nodes = ["Back Surface", "Back Frame", "Seat Surface", "Seat Frame", 
            "Footrest", "Chair Base"]
LEAF_nodes = ["Head Rest", "Head Connector", "Back Hard Surface",
              "Back Soft Surface", "Back Support", "Back Holistic Frame",
              "Back Surface Vertical Bar", "Back Surface Horizontal Bar",
              "Back Surface Slant Bar", "Back Frame Horizontal Bar", 
              "Back Frame Slant Bar", "Back Frame Vertical Bar", "Back Complex Decoration",
              "Back Connector", "Armrest Hard Surface", "Armrest Soft Surface",
              "Arm Horizontal Bar", "Arm Vertical Bar", "Arm Slant Bar", 
              "Arm Holistic Frame", "Sofa-style Arm", "Arm Writing Table",
              "Arm Connector", "Seat Hard Surface", "Seat Soft Surface",
              "Seat Support", "Seat Surface Vertical Bar",
              "Seat Surface Horizontal Bar", "Seat Surface Slant Bar",
              "Seat Frame Vertical Bar", "Seat Frame Horizontal Bar",
              "Seat Frame Slant Bar", "Seat Holistic Frame", "Pillow",
              "Footrest Connector", "Bar Stretcher", "Circular Stretcher",
              "Runner", "Rocker", "Seat Connector", "Central Support",
              "Lever", "Knob", "Button", "Mounting Plane", "Footrest Bar",
              "Footrest Circle", "Pedestal", "Short Leg", "Ground Surface",
              "Side Surface", "Leg", "Foot", "Brake", "Caster Yoke", "Caster Stem",
              "Mounting Plate", "Wheel"]

def parse_chair(chair):
  error_flag = False
  Tree_Layers = []
  Dest_Layers = []
  Stell_Src = []
  Stell_Dest = []
  def compute_tree(root):
    for item in root:
      current = item['text']
      #print(current)
      childrens = item.get('children')
      #print(childrens)
      # AND or OR node  
      if (childrens):
        source = str(item['name']) + "_" + str(item['id'])
        Tree_Layers.append(source)
        stell_source = str(item['text'])
        Stell_Src.append(stell_source)
        this_children = childrens
        Dest_Layers_current = []
        Stell_Dest_current = []
        for i in this_children:
          current_i = (str(i['name']) + "_" + str(i['id']))
          #print(current_i)
          Dest_Layers_current.append(current_i)
          Stell_Dest_current.append(str(i['text']))
        Dest_Layers.append(Dest_Layers_current)
        Stell_Dest.append(Stell_Dest_current)
        # AND node
        if (current in AND_nodes):
          #print("AND Node")
          dict_entry = str(item['name']) + "_" + str(item['id'])
          #print(dict_entry)
          compute_tree(childrens)

        elif (current in OR_nodes):
          #print("OR Node")
          dict_entry = str(item['name']) + "_" + str(item['id'])
          #print(dict_entry)
          compute_tree(childrens)

        else:
          error_flag = True
          return

      else:

        # LEAF node
        if (current in LEAF_nodes):
          #print("LEAF Node")
          dict_entry = str(item['name']) + "_" + str(item['id'])
          #print(dict_entry)
          #compute_tree(childrens)

        else:
          print("Error in classification.")
          return

  compute_tree(chair)
  if (error_flag == True):
    return [[],[]]

  #print([Tree_Layers, Dest_Layers])
  return [Stell_Src, Stell_Dest]

#print(parse_chair(result_data))