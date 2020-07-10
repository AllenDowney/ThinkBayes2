import sys

from Filist import Filist

def main(name, filename, *argv):
    # print the contents of the given file
    ft = Filist(filename)
    ft.sub_lines(r'<programlisting>plasTeXpython', r'<programlisting language="python">')
    ft.sub_lines(r'plasTeXangle', r'&lt;&gt;')

    # label the last chapters as appendices
    #i, match = ft.search_lines('<chapter id="tools">')
    #ft.sub_lines(r'<chapter', r'<appendix', start=i)
    #ft.sub_lines(r'</chapter', r'</appendix', start=i+1)

    ft.sub_lines(r'<emphasis role="bold">feedback@greenteapress.com</emphasis>',
                 r'<phrase role="keep-together"><emphasis role="bold">feedback@greenteapress.com</emphasis></phrase>')

    print ft

if __name__ == '__main__':
    main(*sys.argv)
