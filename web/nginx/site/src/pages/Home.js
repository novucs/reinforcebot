import React, {Component} from 'react'
import {
  Button,
  Container,
  Divider,
  Grid,
  Header,
  Icon,
  Image,
  List,
  Menu,
  Responsive,
  Segment,
  Sidebar,
  Visibility,
} from 'semantic-ui-react'
import whiteLogo from "../white-logo.svg";
import problemImg from "../problem.png";
import solutionImg from "../solution.png";

const getWidth = () => {
  const isSSR = typeof window === 'undefined';
  return isSSR ? Responsive.onlyTablet.minWidth : window.innerWidth
};

const HomepageHeading = ({mobile}) => (
  <Container text>
    <img
      style={{
        marginBottom: mobile ? '`1.5em' : '3em',
        marginTop: mobile ? '6em' : '12em'
      }}
      src={whiteLogo}
      alt="logo"
      className="image"
    />
    <Button primary size='huge'>
      Get Started
      <Icon name='right arrow'/>
    </Button>
  </Container>
);

class DesktopContainer extends Component {
  state = {};

  hideFixedMenu = () => this.setState({fixed: false});
  showFixedMenu = () => this.setState({fixed: true});

  render() {
    const {children} = this.props;
    const {fixed} = this.state;

    return (
      <Responsive getWidth={getWidth} minWidth={Responsive.onlyTablet.minWidth}>
        <Visibility
          once={false}
          onBottomPassed={this.showFixedMenu}
          onBottomPassedReverse={this.hideFixedMenu}
        >
          <Segment
            inverted
            textAlign='center'
            style={{minHeight: 700, padding: '1em 0em'}}
            vertical
          >
            <Menu
              fixed={fixed ? 'top' : null}
              inverted={!fixed}
              pointing={!fixed}
              secondary={!fixed}
              size='large'
            >
              <Container>
                <Menu.Item as='a' active>
                  Home
                </Menu.Item>
                <Menu.Item position='right'>
                  <Button as='a' href='/signin' inverted={!fixed}>
                    Sign in
                  </Button>
                  <Button as='a' href='/signup' inverted={!fixed} primary={fixed} style={{marginLeft: '0.5em'}}>
                    Sign Up
                  </Button>
                </Menu.Item>
              </Container>
            </Menu>
            <HomepageHeading/>
          </Segment>
        </Visibility>

        {children}
      </Responsive>
    )
  }
}

class MobileContainer extends Component {
  state = {};

  handleSidebarHide = () => this.setState({sidebarOpened: false});

  handleToggle = () => this.setState({sidebarOpened: true});

  render() {
    const {children} = this.props;
    const {sidebarOpened} = this.state;

    return (
      <Responsive
        as={Sidebar.Pushable}
        getWidth={getWidth}
        maxWidth={Responsive.onlyMobile.maxWidth}
      >
        <Sidebar
          as={Menu}
          animation='push'
          inverted
          onHide={this.handleSidebarHide}
          vertical
          visible={sidebarOpened}
        >
          <Menu.Item as='a' active>
            Home
          </Menu.Item>
          <Menu.Item as='a'>Sign in</Menu.Item>
          <Menu.Item as='a'>Sign Up</Menu.Item>
        </Sidebar>

        <Sidebar.Pusher dimmed={sidebarOpened}>
          <Segment
            inverted
            textAlign='center'
            style={{minHeight: 350, padding: '1em 0em'}}
            vertical
          >
            <Container>
              <Menu inverted pointing secondary size='large'>
                <Menu.Item onClick={this.handleToggle}>
                  <Icon name='sidebar'/>
                </Menu.Item>
                <Menu.Item position='right'>
                  <Button as='a' inverted>
                    Sign in
                  </Button>
                  <Button as='a' inverted style={{marginLeft: '0.5em'}}>
                    Sign Up
                  </Button>
                </Menu.Item>
              </Menu>
            </Container>
            <HomepageHeading mobile/>
          </Segment>

          {children}
        </Sidebar.Pusher>
      </Responsive>
    )
  }
}

const ResponsiveContainer = ({children}) => (
  <div>
    <DesktopContainer>{children}</DesktopContainer>
    <MobileContainer>{children}</MobileContainer>
  </div>
);

class Footer extends Component {
  render() {
    return (
      <Segment inverted vertical style={{padding: '5em 0em'}}>
        <Container>
          <Grid divided inverted stackable>
            <Grid.Row>
              <Grid.Column width={3}>
                <Header inverted as='h4' content='About'/>
                <List link inverted>
                  <List.Item as='a' href='start'>Quickstart</List.Item>
                  <List.Item as='a' href='mailto:william2.randall@live.uwe.ac.uk'>Contact Us</List.Item>
                </List>
              </Grid.Column>
              <Grid.Column width={7}>
                <Header as='h4' inverted>
                  ReinforceBot
                </Header>
                <p>
                  Automates the creation of software agents that can interact with the desktop environment.
                </p>
              </Grid.Column>
            </Grid.Row>
          </Grid>
        </Container>
      </Segment>
    )
  }
}

const HomepageLayout = () => (
  <ResponsiveContainer>
    <Segment style={{padding: '8em 0em'}} vertical>
      <Grid container stackable verticalAlign='middle'>
        <Grid.Row>
          <Grid.Column width={8}>
            <Header as='h3' style={{fontSize: '2em'}}>
              Ever been tired of repetitive tasks?
            </Header>
            <p style={{fontSize: '1.33em'}}>
              Tasks on your computer may at time to time become repetitive.

              The only two solutions around include creating macros, and developing an agent to automate the workflow.

              However, macros are brittle, and hiring a developer to create an agent can be expensive.
            </p>
          </Grid.Column>
          <Grid.Column floated='right' width={6}>
            <Image bordered rounded size='medium' src={problemImg}/>
          </Grid.Column>
        </Grid.Row>
      </Grid>
    </Segment>
    <Segment style={{padding: '8em 0em'}} vertical>
      <Grid container stackable verticalAlign='middle'>
        <Grid.Row>
          <Grid.Column floated='left' width={6}>
            <Image bordered rounded size='medium' src={solutionImg}/>
          </Grid.Column>
          <Grid.Column width={8}>
            <Header as='h3' style={{fontSize: '2em'}}>
              ReinforceBot automates your tasks
            </Header>
            <p style={{fontSize: '1.33em'}}>
              ReinforceBot enables you to create custom agents with no prior
              development experience required. These agents will be happy to
              automate your task for you, and are even capable of playing games!
            </p>
          </Grid.Column>
        </Grid.Row>
      </Grid>
    </Segment>
    <Footer />
  </ResponsiveContainer>
);

export default HomepageLayout
